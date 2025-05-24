import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import re

def clean_text(text):
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 .,?!\'\"]', '', text)
    return text.strip().lower()

# Paths
DATA_DIR = '.'
MODEL_DIR = '../../streamlit_app/models/'
real_files = ['True.csv', 'synthetic_true.csv']
fake_files = ['Fake.csv', 'synthetic_fake.csv']

# Load data
real_dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in real_files if os.path.exists(os.path.join(DATA_DIR, f))]
fake_dfs = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in fake_files if os.path.exists(os.path.join(DATA_DIR, f))]

real_df = pd.concat(real_dfs, ignore_index=True) if real_dfs else pd.DataFrame()
fake_df = pd.concat(fake_dfs, ignore_index=True) if fake_dfs else pd.DataFrame()

real_df['label'] = 1
fake_df['label'] = 0

df = pd.concat([real_df, fake_df], ignore_index=True)
print(f'Total samples: {len(df)} | Real: {len(real_df)} | Fake: {len(fake_df)}')

df = df.drop_duplicates(subset='text').sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Clean and filter
print('Cleaning text...')
df['text'] = df['text'].apply(clean_text)
df = df[df['text'].str.len() > 30]
df = df.dropna(subset=['text'])
print('After cleaning:', len(df))

# Split data
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)
print(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(train_df['text'])
X_val = vectorizer.transform(val_df['text'])
X_test = vectorizer.transform(test_df['text'])
y_train = train_df['label']
y_val = val_df['label']
y_test = test_df['label']

# Logistic Regression
print('\nTraining Logistic Regression...')
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
val_preds = lr.predict(X_val)
print('Logistic Regression Validation:')
print(classification_report(y_val, val_preds))
print('Confusion matrix:')
print(confusion_matrix(y_val, val_preds))

# Random Forest
print('\nTraining Random Forest...')
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
val_preds_rf = rf.predict(X_val)
print('Random Forest Validation:')
print(classification_report(y_val, val_preds_rf))
print('Confusion matrix:')
print(confusion_matrix(y_val, val_preds_rf))

# Test Set Evaluation (using Logistic Regression)
test_preds = lr.predict(X_test)
print('\nLogistic Regression Test:')
print(classification_report(y_test, test_preds))
print('Confusion matrix:')
print(confusion_matrix(y_test, test_preds))

# Save models and vectorizer
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(lr, os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'))
joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest_model.pkl'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
print('\nModels and vectorizer saved to', MODEL_DIR)
