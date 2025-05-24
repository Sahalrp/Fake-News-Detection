import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create cells
cells = [
    nbf.v4.new_markdown_cell('''# Fake News Detection - Basic Models Implementation

This notebook implements basic machine learning models for fake news detection using our engineered features.'''),
    
    nbf.v4.new_code_cell('''# Part 1: Setup and Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Configure plot settings
sns.set_theme(style='whitegrid')  # This is the correct way to set seaborn style
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12'''),
    
    nbf.v4.new_code_cell('''# Part 2: Load and Examine Data
# Load the preprocessed dataset
df = pd.read_csv('../3_Data_Analysis/news_with_sentiment.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\\nColumns:", df.columns.tolist())
print("\\nSample of numerical features:")
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
print(df[numerical_features].describe())'''),

    nbf.v4.new_code_cell('''# Part 3: Feature Selection and Preparation
# Select features for modeling
feature_columns = [
    # Text length features
    'text_length', 'title_length',
    
    # Stylometric features
    'exclamation_density', 'question_density', 'quotes_density',
    'capitalized_ratio', 'all_caps_ratio', 'special_chars_density',
    
    # Sentence structure features
    'avg_sentence_length', 'sentence_length_std', 'num_sentences',
    
    # Sentiment features
    'polarity', 'subjectivity', 'sentiment_std',
    'max_polarity', 'min_polarity'
]

# Prepare feature matrix X and target variable y
X = df[feature_columns]
y = df['label']

# Check for missing values
print("Missing values in features:")
print(X.isnull().sum())

# Convert target labels to numeric
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Display feature statistics after scaling
print("\\nFeature statistics after scaling:")
print(X_scaled.describe())'''),

    nbf.v4.new_code_cell('''# Part 4: Feature Analysis
# Create correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(X_scaled.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Plot feature distributions
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
axes = axes.ravel()

for idx, column in enumerate(X_scaled.columns):
    if idx < len(axes):
        sns.histplot(data=X_scaled, x=column, hue=y, ax=axes[idx], bins=30)
        axes[idx].set_title(f'Distribution of {column}')

plt.tight_layout()
plt.show()'''),

    nbf.v4.new_code_cell('''# Part 5: Data Splitting
# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Display class distribution
print("\\nClass distribution in training set:")
print(pd.Series(y_train).value_counts(normalize=True))
print("\\nClass distribution in test set:")
print(pd.Series(y_test).value_counts(normalize=True))'''),

    nbf.v4.new_markdown_cell('''## Model Implementation and Evaluation'''),

    nbf.v4.new_code_cell('''# Part 6: Logistic Regression Model
# Train logistic regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Print model evaluation
print("Logistic Regression Results:")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_lr, 
                          target_names=['Real', 'Fake']))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': np.abs(lr_model.coef_[0])
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance - Logistic Regression')
plt.show()'''),

    nbf.v4.new_code_cell('''# Part 7: Random Forest Model
# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Print model evaluation
print("Random Forest Results:")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_rf,
                          target_names=['Real', 'Fake']))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance
feature_importance_rf = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
})
feature_importance_rf = feature_importance_rf.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_rf, x='importance', y='feature')
plt.title('Feature Importance - Random Forest')
plt.show()'''),

    nbf.v4.new_markdown_cell('''## Model Comparison and Analysis

We've implemented two models:
1. Logistic Regression: A simple linear model that serves as our baseline
2. Random Forest: A more complex ensemble model that can capture non-linear relationships

Key points to analyze:
- Compare accuracy, precision, recall, and F1-scores
- Look at confusion matrices to understand error patterns
- Compare feature importance between models
- Consider model interpretability vs. performance tradeoffs'''),

    nbf.v4.new_code_cell('''# Part 8: Model Comparison
# Compare feature importance rankings
lr_features = feature_importance['feature'].tolist()
rf_features = feature_importance_rf['feature'].tolist()

print("Top 5 Most Important Features:")
print("\\nLogistic Regression:")
for idx, feature in enumerate(lr_features[:5], 1):
    print(f"{idx}. {feature}")

print("\\nRandom Forest:")
for idx, feature in enumerate(rf_features[:5], 1):
    print(f"{idx}. {feature}")

# Plot feature importance comparison
plt.figure(figsize=(12, 6))
feature_comparison = pd.DataFrame({
    'feature': feature_columns,
    'Logistic Regression': lr_model.coef_[0],
    'Random Forest': rf_model.feature_importances_
})

feature_comparison_melted = pd.melt(
    feature_comparison, 
    id_vars=['feature'], 
    var_name='Model', 
    value_name='Importance'
)

sns.barplot(data=feature_comparison_melted, x='Importance', y='feature', hue='Model')
plt.title('Feature Importance Comparison Between Models')
plt.show()''')
]

# Add the cells to the notebook
nb['cells'] = cells

# Write the notebook to a file
with open('01_Basic_Models.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 