from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
from typing import Dict, Any

class BaselineModels:
    """Implementation of baseline models for fake news detection."""
    
    def __init__(self):
        self.models = {
            'logistic_regression': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000)),
                ('classifier', LogisticRegression(random_state=42))
            ]),
            'random_forest': Pipeline([
                ('vectorizer', TfidfVectorizer(max_features=5000)),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        }
    
    def train_and_evaluate(self, X_train, y_train, cv=5) -> Dict[str, Any]:
        """Train and evaluate models using cross-validation."""
        results = {}
        
        for name, model in self.models.items():
            # Perform cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            results[name] = {
                'mean_accuracy': scores.mean(),
                'std_accuracy': scores.std(),
                'cv_scores': scores
            }
            
            # Fit the model on the entire training set
            model.fit(X_train, y_train)
        
        return results
    
    def predict(self, X, model_name: str):
        """Make predictions using the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        return self.models[model_name].predict(X)
    
    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Get feature importance for the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        vectorizer = model.named_steps['vectorizer']
        classifier = model.named_steps['classifier']
        
        if model_name == 'logistic_regression':
            importance = np.abs(classifier.coef_[0])
        else:  # random_forest
            importance = classifier.feature_importances_
        
        feature_importance = dict(zip(
            vectorizer.get_feature_names_out(),
            importance
        ))
        
        return dict(sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )) 