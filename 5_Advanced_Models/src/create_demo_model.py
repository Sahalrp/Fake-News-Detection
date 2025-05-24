import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Sample data for demo
sample_data = {
    'text': [
        "Breaking: Scientific study confirms climate change impact on global temperatures",
        "SHOCKING: Aliens spotted in local grocery store buying all the toilet paper!!!",
        "New research shows benefits of regular exercise on mental health",
        "OMG! Pizza found to cure all diseases overnight! Doctors hate this!",
        "Economic report indicates steady growth in manufacturing sector",
        "UNBELIEVABLE: Government hiding giant dragons in secret underground bases!!!"
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 for real, 0 for fake
}

# Create and save demo model
def create_demo_model():
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Create and fit vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    
    # Create and fit model
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X, y)
    
    # Save model and vectorizer
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open(os.path.join(model_dir, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    print("Demo model created successfully!")

if __name__ == "__main__":
    create_demo_model()
