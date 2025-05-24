import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy as np
from typing import Dict, List, Union, Any, Optional
from tqdm import tqdm

from .base_model import BaseModel

class NewsDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer=None, max_length: int = 512):
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
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

class TransformerModel(BaseModel):
    """BERT-based model for fake news detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BERT model.
        
        Args:
            config: Dictionary containing model configuration parameters
                - model_name: Name of the pretrained model to use
                - max_length: Maximum sequence length
                - batch_size: Batch size for training
                - learning_rate: Learning rate for optimization
                - num_epochs: Number of training epochs
        """
        super().__init__(config)
        
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.num_epochs = config.get('num_epochs', 3)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the model with base configuration."""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            # Add default configuration for fake news detection
            id2label={0: "REAL", 1: "FAKE"},
            label2id={"REAL": 0, "FAKE": 1}
        ).to(self.device)
        
    def preprocess(self, text: Union[str, List[str]]) -> NewsDataset:
        """Preprocess the input text data."""
        if isinstance(text, str):
            text = [text]
        return NewsDataset(text, tokenizer=self.tokenizer, max_length=self.max_length)
    
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str] = None, val_labels: List[int] = None) -> Dict[str, float]:
        """Train the model on the given dataset."""
        train_dataset = NewsDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            self.max_length
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(self.num_epochs):
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.num_epochs}')
            epoch_loss = 0
            
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
                val_metrics = self.evaluate(val_texts, val_labels)
                print(f"Validation metrics: {val_metrics}")
        
        return {'avg_train_loss': total_loss / self.num_epochs}
    
    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Make predictions on the input texts."""
        proba = self.predict_proba(texts)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Get probability estimates for predictions."""
        if isinstance(texts, str):
            texts = [texts]
            
        dataset = self.preprocess(texts)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        self.model.eval()
        probas = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                proba = torch.softmax(logits, dim=1)
                probas.append(proba.cpu().numpy())
        
        return np.vstack(probas)
    
    def save(self, path: str):
        """Save the model to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    def load(self, path: str):
        """Load the model from disk."""
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path) 