"""
Utility functions for the Fake News Detection Streamlit app.
"""

import os
import re
import sys
import pandas as pd
import joblib
from textblob import TextBlob
import streamlit as st

# Add the parent directory to the path so we can import the llm_verification module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Text preprocessing
def clean_text(text):
    """
    Clean and normalize text for consistent preprocessing.

    Args:
        text (str): Input text

    Returns:
        str: Cleaned text
    """
    if pd.isna(text) or text == "":
        return ""

    text = str(text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-alphanumeric characters except punctuation
    text = re.sub(r'[^a-zA-Z0-9 .,?!\'\"]', '', text)
    # Convert to lowercase and strip
    return text.strip().lower()

# Feature extraction
def extract_stylometric_features(text):
    """
    Extract stylometric features from text.

    Args:
        text (str): Input text

    Returns:
        dict: Dictionary of stylometric features
    """
    if pd.isna(text) or text == "":
        return {
            'text_length': 0,
            'avg_sentence_length': 0,
            'exclamation_density': 0,
            'question_density': 0,
            'quotes_density': 0,
            'caps_ratio': 0
        }

    # Text length
    text_length = len(text)

    # Sentence features
    sentences = text.split('.')
    num_sentences = len(sentences)
    words = text.split()
    avg_sentence_length = len(words) / max(num_sentences, 1)

    # Punctuation features
    exclamation_density = text.count('!') / max(text_length, 1)
    question_density = text.count('?') / max(text_length, 1)
    quotes_density = (text.count('"') + text.count("'")) / max(text_length, 1)

    # Capitalization
    caps_words = sum(1 for word in words if word.isupper())
    caps_ratio = caps_words / max(len(words), 1)

    return {
        'text_length': text_length,
        'avg_sentence_length': avg_sentence_length,
        'exclamation_density': exclamation_density,
        'question_density': question_density,
        'quotes_density': quotes_density,
        'caps_ratio': caps_ratio
    }

def extract_sentiment_features(text):
    """
    Extract sentiment features from text.

    Args:
        text (str): Input text

    Returns:
        dict: Dictionary of sentiment features
    """
    if pd.isna(text) or text == "":
        return {
            'sentiment_polarity': 0,
            'sentiment_subjectivity': 0
        }

    blob = TextBlob(text)
    return {
        'sentiment_polarity': blob.sentiment.polarity,
        'sentiment_subjectivity': blob.sentiment.subjectivity
    }

# Model loading
@st.cache_resource
def load_models(root_dir):
    """
    Load machine learning models.

    Args:
        root_dir (str): Root directory path

    Returns:
        tuple: (lr_model, rf_model, vectorizer) or (None, None, None) if loading fails
    """
    try:
        model_dir = os.path.join(root_dir, 'models')
        if not os.path.exists(model_dir):
            st.error(f"Models directory not found at {model_dir}")
            return None, None, None

        lr_path = os.path.join(model_dir, 'logistic_regression_model.pkl')
        rf_path = os.path.join(model_dir, 'random_forest_model.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')

        if not all(os.path.exists(p) for p in [lr_path, rf_path, vectorizer_path]):
            st.error("Some model files are missing!")
            return None, None, None

        lr_model = joblib.load(lr_path)
        rf_model = joblib.load(rf_path)
        vectorizer = joblib.load(vectorizer_path)
        return lr_model, rf_model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Data loading
@st.cache_data
def load_data(root_dir):
    """
    Load dataset.

    Args:
        root_dir (str): Root directory path

    Returns:
        pandas.DataFrame or None: Loaded dataset or None if loading fails
    """
    try:
        file_path = os.path.join(root_dir, 'data', 'news_with_features.csv')
        if not os.path.exists(file_path):
            st.error(f"Dataset not found at {file_path}")
            return None
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# System status check
def check_system_status(root_dir):
    """
    Check system status.

    Args:
        root_dir (str): Root directory path

    Returns:
        dict: Status of models and data
    """
    status = {
        "models": {
            "path": os.path.join(root_dir, "models"),
            "files": ["logistic_regression_model.pkl", "random_forest_model.pkl", "tfidf_vectorizer.pkl"]
        },
        "data": {
            "path": os.path.join(root_dir, "data"),
            "files": ["news_with_features.csv"]
        }
    }

    results = {}

    # Check models
    model_status = True
    if not os.path.exists(status["models"]["path"]):
        model_status = False
    else:
        for file in status["models"]["files"]:
            if not os.path.exists(os.path.join(status["models"]["path"], file)):
                model_status = False
                break
    results["models"] = model_status

    # Check data
    data_status = True
    if not os.path.exists(status["data"]["path"]):
        data_status = False
    else:
        for file in status["data"]["files"]:
            if not os.path.exists(os.path.join(status["data"]["path"], file)):
                data_status = False
                break
    results["data"] = data_status

    return results

# LLM verification
@st.cache_resource
def load_llm_verifier(model_name="deepseek-r1:7b", temperature=0.3):
    """
    Load the LLM-based news verifier.

    Args:
        model_name (str): Name of the LLM model to use
        temperature (float): Temperature for LLM generation

    Returns:
        NewsVerifier or None: Loaded verifier or None if loading fails
    """
    try:
        # Import here to avoid circular imports
        from llm_verification import NewsVerifier

        verifier = NewsVerifier(
            model_name=model_name,
            temperature=temperature
        )

        # Check if Ollama is available
        if not verifier.check_availability():
            st.warning("Ollama is not running. Please start the Ollama application.")
            return None

        return verifier
    except ImportError:
        st.error("Could not import the LLM verification module. Please make sure it's installed.")
        return None
    except Exception as e:
        st.error(f"Error loading LLM verifier: {str(e)}")
        return None

def check_ollama_status():
    """
    Check if Ollama is running and get available models.

    Returns:
        dict: Dictionary with status information
            - running: True if Ollama is running, False otherwise
            - models: List of available models if running
            - error: Error message if any
    """
    status = {
        "running": False,
        "models": []
    }

    try:
        # Import here to avoid circular imports
        from llm_verification import OllamaClient

        client = OllamaClient()
        status["running"] = client.check_availability()

        if status["running"]:
            # Get available models
            models = client.list_models()
            status["models"] = [model.get("name", "").split(":")[0] for model in models]

    except ImportError as e:
        status["error"] = "LLM verification module not found"
    except Exception as e:
        status["error"] = str(e)

    return status
