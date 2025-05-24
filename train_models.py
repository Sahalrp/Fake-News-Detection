import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('news_with_features.csv')

# Prepare the data
X = df['text']
y = (df['label'] == 'Real').astype(int)  # Convert to binary (0 for Fake, 1 for Real)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train TF-IDF vectorizer
print("Training TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression
print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_vec, y_train)

# Train Random Forest
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_vec, y_train)

# Evaluate models
print("\nLogistic Regression Performance:")
lr_pred = lr_model.predict(X_test_vec)
print(classification_report(y_test, lr_pred, target_names=['Fake', 'Real']))

print("\nRandom Forest Performance:")
rf_pred = rf_model.predict(X_test_vec)
print(classification_report(y_test, rf_pred, target_names=['Fake', 'Real']))

# Save models
print("\nSaving models...")
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("Models saved successfully!")
print("Model files:")
print("- models/logistic_regression_model.pkl")
print("- models/random_forest_model.pkl")
print("- models/tfidf_vectorizer.pkl") 