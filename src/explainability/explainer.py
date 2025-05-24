import numpy as np
from typing import List, Dict, Any, Callable
import shap
import lime.lime_text
from transformers import AutoTokenizer

class ModelExplainer:
    """Class for generating explanations for model predictions."""
    
    def __init__(self, predict_fn: Callable, tokenizer: AutoTokenizer = None):
        """
        Initialize the explainer.
        
        Args:
            predict_fn: Model prediction function that takes text input and returns probabilities
            tokenizer: Tokenizer for transformer models (required for SHAP explanations)
        """
        self.predict_fn = predict_fn
        self.tokenizer = tokenizer
        self.lime_explainer = lime.lime_text.LimeTextExplainer(
            class_names=['Real', 'Fake']
        )
        
    def explain_with_lime(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Generate LIME explanation for a single text input.
        
        Args:
            text: Input text to explain
            num_features: Number of features to include in the explanation
            
        Returns:
            Dictionary containing explanation details
        """
        exp = self.lime_explainer.explain_instance(
            text,
            self.predict_fn,
            num_features=num_features
        )
        
        # Get feature importance scores
        feature_importance = dict(exp.as_list())
        
        # Get prediction probabilities
        probs = self.predict_fn([text])[0]
        predicted_class = 'Fake' if probs[1] > probs[0] else 'Real'
        
        return {
            'prediction': predicted_class,
            'probability': float(max(probs)),
            'feature_importance': feature_importance,
            'explanation_html': exp.as_html()
        }
    
    def explain_with_shap(self, texts: List[str], num_samples: int = 100) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a batch of texts.
        
        Args:
            texts: List of input texts to explain
            num_samples: Number of background samples to use
            
        Returns:
            Dictionary containing explanation details
        """
        # Create a word-level explainer using SHAP's DeepExplainer
        explainer = shap.Explainer(
            model=self.predict_fn,
            masker=shap.maskers.Text(self.tokenizer)
        )
        
        # Calculate SHAP values
        shap_values = explainer(texts[:num_samples])
        
        # Get predictions
        predictions = self.predict_fn(texts)
        predicted_classes = ['Fake' if p[1] > p[0] else 'Real' for p in predictions]
        
        # Prepare visualization data
        explanations = []
        for idx, text in enumerate(texts[:num_samples]):
            word_importance = dict(zip(
                shap_values.data[idx],
                shap_values.values[idx]
            ))
            
            explanations.append({
                'text': text,
                'prediction': predicted_classes[idx],
                'probability': float(max(predictions[idx])),
                'word_importance': word_importance
            })
        
        return {
            'explanations': explanations,
            'shap_values': shap_values
        }
    
    def get_global_feature_importance(self, texts: List[str], num_samples: int = 100) -> Dict[str, float]:
        """
        Calculate global feature importance across multiple texts.
        
        Args:
            texts: List of input texts
            num_samples: Number of samples to use
            
        Returns:
            Dictionary mapping features to their importance scores
        """
        # Use SHAP for global feature importance
        shap_output = self.explain_with_shap(texts[:num_samples])
        shap_values = shap_output['shap_values']
        
        # Calculate mean absolute SHAP values for each feature
        global_importance = {}
        for idx, feature in enumerate(shap_values.data[0]):
            importance = np.abs(shap_values.values[:, idx]).mean()
            global_importance[feature] = float(importance)
        
        # Sort by importance
        return dict(sorted(
            global_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )) 