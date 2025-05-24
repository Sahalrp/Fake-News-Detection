from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Union, Any
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BaseModel(ABC):
    """Abstract base class for all fake news detection models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base model.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        self.config = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def preprocess(self, text: Union[str, List[str]]) -> Any:
        """
        Preprocess the input text data.
        
        Args:
            text: Input text or list of texts to preprocess
            
        Returns:
            Preprocessed text in the format required by the model
        """
        pass
    
    @abstractmethod
    def train(self, train_texts: List[str], train_labels: List[int],
              val_texts: List[str] = None, val_labels: List[int] = None) -> Dict[str, float]:
        """
        Train the model on the given dataset.
        
        Args:
            train_texts: List of training text samples
            train_labels: List of training labels (0 for real, 1 for fake)
            val_texts: Optional list of validation text samples
            val_labels: Optional list of validation labels
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Make predictions on the input texts.
        
        Args:
            texts: Input text or list of texts to classify
            
        Returns:
            numpy array of predictions (0 for real, 1 for fake)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Get probability estimates for predictions.
        
        Args:
            texts: Input text or list of texts to classify
            
        Returns:
            numpy array of shape (n_samples, 2) containing probability estimates
        """
        pass
    
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the model's performance on a test set.
        
        Args:
            texts: List of test text samples
            labels: List of test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        predictions = self.predict(texts)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions)
        }
    
    def save(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path where to save the model
        """
        raise NotImplementedError("Save method must be implemented by child class")
    
    def load(self, path: str):
        """
        Load the model from disk.
        
        Args:
            path: Path from where to load the model
        """
        raise NotImplementedError("Load method must be implemented by child class") 