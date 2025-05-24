import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

class NewsDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
            
        return item

class BERTModel:
    """BERT-based model for fake news detection."""
    
    def __init__(self, model_name='bert-base-uncased', max_length=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.max_length = max_length
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str] = None, val_labels: List[int] = None,
              batch_size: int = 8, num_epochs: int = 3,
              learning_rate: float = 2e-5) -> Dict[str, Any]:
        """Train the model."""
        # Create datasets
        train_dataset = NewsDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.max_length
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        
        # Training loop
        self.model.train()
        total_loss = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            total_loss += epoch_loss / len(train_loader)
            
            if val_texts is not None and val_labels is not None:
                val_accuracy = self.evaluate(val_texts, val_labels)
                print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        return {'avg_train_loss': total_loss / num_epochs}
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """Make predictions on the input texts."""
        dataset = NewsDataset(texts, tokenizer=self.tokenizer, max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=8)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def evaluate(self, texts: List[str], labels: List[int]) -> float:
        """Evaluate the model on the given dataset."""
        predictions = self.predict(texts)
        return np.mean(predictions == labels)
    
    def save(self, path: str):
        """Save the model to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load the model from disk."""
        self.model = BertForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(path) 