from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
from typing import Dict, Any

class SVMModel:
    """Support Vector Machine implementation for fake news detection."""
    
    def __init__(self):
        self.pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000)),
            ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
        ])
        
        self.param_grid = {
            'vectorizer__max_features': [3000, 5000, 7000],
            'vectorizer__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': ['scale', 'auto', 0.1, 1]
        }
    
    def train_with_grid_search(self, X_train, y_train, cv=5) -> Dict[str, Any]:
        """Train model with grid search for hyperparameter optimization."""
        grid_search = GridSearchCV(
            self.pipeline,
            self.param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.pipeline = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates."""
        return self.pipeline.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on SVM coefficients."""
        vectorizer = self.pipeline.named_steps['vectorizer']
        classifier = self.pipeline.named_steps['classifier']
        
        if classifier.kernel == 'linear':
            importance = np.abs(classifier.coef_[0])
        else:
            # For non-linear kernels, use feature weights based on support vectors
            importance = np.abs(np.sum(classifier.dual_coef_[0].reshape(1, -1), axis=1))
        
        feature_importance = dict(zip(
            vectorizer.get_feature_names_out(),
            importance
        ))
        
        return dict(sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )) 