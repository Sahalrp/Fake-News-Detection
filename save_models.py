import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import joblib
import os

def extract_features(text):
    if pd.isna(text) or text == "":
        return {
            'text_length': 0,
            'avg_sentence_length': 0,
            'exclamation_density': 0,
            'question_density': 0,
            'quotes_density': 0,
            'caps_ratio': 0,
            'sentiment_polarity': 0,
            'sentiment_subjectivity': 0
        }

    # Stylometric features
    text_str = str(text)
    text_length = len(text_str)  # Character count for consistency with Streamlit app
    total_words = len(text_str.split())
    sentences = text_str.split('.')
    avg_sentence_length = total_words / max(len(sentences), 1)

    exclamation_density = text_str.count('!') / max(text_length, 1)
    question_density = text_str.count('?') / max(text_length, 1)
    quotes_density = (text_str.count('"') + text_str.count("'")) / max(text_length, 1)

    words = text_str.split()
    caps_words = sum(1 for word in words if word.isupper())
    caps_ratio = caps_words / max(len(words), 1)

    # Sentiment features
    blob = TextBlob(text_str)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity

    return {
        'text_length': text_length,
        'avg_sentence_length': avg_sentence_length,
        'exclamation_density': exclamation_density,
        'question_density': question_density,
        'quotes_density': quotes_density,
        'caps_ratio': caps_ratio,
        'sentiment_polarity': sentiment_polarity,
        'sentiment_subjectivity': sentiment_subjectivity
    }

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load and combine datasets
print("Loading datasets...")
true_df = pd.read_csv('2_Data_Collection/data/True.csv')
fake_df = pd.read_csv('2_Data_Collection/data/Fake.csv')

# Add labels
true_df['label'] = 'Real'
fake_df['label'] = 'Fake'

# Combine datasets
df = pd.concat([true_df, fake_df], ignore_index=True)

# Extract features
print("Extracting features...")
print("Total articles to process:", len(df))

# Process in batches to show progress
batch_size = 1000
features_list = []

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    batch_features = batch['text'].apply(extract_features)
    features_list.extend(batch_features)
    print(f"Processed {min(i+batch_size, len(df))} out of {len(df)} articles")

features_df = pd.DataFrame(features_list)

# Combine features with original dataset
df = pd.concat([df, features_df], axis=1)

# Save the dataset with features
print("Saving dataset with features...")
df.to_csv('news_with_features.csv', index=False)

# Prepare features for training
feature_columns = [
    'text_length', 'avg_sentence_length', 'exclamation_density',
    'question_density', 'quotes_density', 'caps_ratio',
    'sentiment_polarity', 'sentiment_subjectivity'
]
X = df[feature_columns]
y = (df['label'] == 'Real').astype(int)  # Convert to binary (0 for Fake, 1 for Real)

# Train Logistic Regression
print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X, y)

# Train Random Forest
print("Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Save models
print("Saving models...")
joblib.dump(lr_model, 'models/logistic_regression_model.pkl')
joblib.dump(rf_model, 'models/random_forest_model.pkl')

print("\nProcess completed successfully!")
print("Dataset with features saved to: news_with_features.csv")
print("Logistic Regression model saved to: models/logistic_regression_model.pkl")
print("Random Forest model saved to: models/random_forest_model.pkl")