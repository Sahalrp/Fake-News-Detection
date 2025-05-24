# Part 1: Setup and Imports
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
plt.rcParams['font.size'] = 12

print("All imports successful!")
print("\nAvailable matplotlib styles:", plt.style.available) 