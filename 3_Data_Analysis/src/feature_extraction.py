"""
Feature extraction module for the Fake News Detection System.
This module contains functions for extracting features from text data.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import gensim
from gensim.models import Word2Vec, FastText
import nltk
from nltk.tokenize import word_tokenize
import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModel
import torch

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TfidfFeatureExtractor:
    """
    Extract TF-IDF features from text data.
    """
    
    def __init__(self, max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95):
        """
        Initialize the TF-IDF feature extractor.
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (tuple): Range of n-grams to consider
            min_df (int or float): Minimum document frequency
            max_df (float): Maximum document frequency
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True
        )
        self.feature_names = None
    
    def fit_transform(self, texts):
        """
        Fit the vectorizer and transform the texts.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            scipy.sparse.csr.csr_matrix: TF-IDF features
        """
        features = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return features
    
    def transform(self, texts):
        """
        Transform the texts using the fitted vectorizer.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            scipy.sparse.csr.csr_matrix: TF-IDF features
        """
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """
        Get the feature names.
        
        Returns:
            list: List of feature names
        """
        return self.feature_names
    
    def save(self, path):
        """
        Save the vectorizer to a file.
        
        Args:
            path (str): Path to save the vectorizer
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    @classmethod
    def load(cls, path):
        """
        Load a vectorizer from a file.
        
        Args:
            path (str): Path to the saved vectorizer
            
        Returns:
            TfidfFeatureExtractor: Loaded feature extractor
        """
        extractor = cls()
        with open(path, 'rb') as f:
            extractor.vectorizer = pickle.load(f)
        extractor.feature_names = extractor.vectorizer.get_feature_names_out()
        return extractor


class Word2VecFeatureExtractor:
    """
    Extract Word2Vec features from text data.
    """
    
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4, sg=1):
        """
        Initialize the Word2Vec feature extractor.
        
        Args:
            vector_size (int): Dimensionality of the word vectors
            window (int): Maximum distance between the current and predicted word
            min_count (int): Ignores all words with total frequency lower than this
            workers (int): Number of worker threads to train the model
            sg (int): Training algorithm: 1 for skip-gram; otherwise CBOW
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model = None
    
    def fit(self, texts):
        """
        Fit the Word2Vec model.
        
        Args:
            texts (list): List of text documents
        """
        # Tokenize texts
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )
    
    def transform(self, texts):
        """
        Transform texts to document vectors by averaging word vectors.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Document vectors
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Tokenize texts
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        
        # Transform to document vectors
        doc_vectors = []
        for tokens in tokenized_texts:
            # Filter tokens that are in the vocabulary
            tokens = [token for token in tokens if token in self.model.wv]
            
            if tokens:
                # Average word vectors
                doc_vector = np.mean([self.model.wv[token] for token in tokens], axis=0)
            else:
                # If no tokens are in the vocabulary, use a zero vector
                doc_vector = np.zeros(self.vector_size)
            
            doc_vectors.append(doc_vector)
        
        return np.array(doc_vectors)
    
    def fit_transform(self, texts):
        """
        Fit the Word2Vec model and transform the texts.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Document vectors
        """
        self.fit(texts)
        return self.transform(texts)
    
    def save(self, path):
        """
        Save the Word2Vec model to a file.
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        self.model.save(path)
    
    @classmethod
    def load(cls, path):
        """
        Load a Word2Vec model from a file.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            Word2VecFeatureExtractor: Loaded feature extractor
        """
        extractor = cls()
        extractor.model = Word2Vec.load(path)
        extractor.vector_size = extractor.model.vector_size
        return extractor


class TransformerFeatureExtractor:
    """
    Extract features from text data using transformer models.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', max_length=512):
        """
        Initialize the transformer feature extractor.
        
        Args:
            model_name (str): Name of the transformer model
            max_length (int): Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Set the model to evaluation mode
        self.model.eval()
    
    def transform(self, texts, batch_size=8):
        """
        Transform texts to embeddings using the transformer model.
        
        Args:
            texts (list): List of text documents
            batch_size (int): Batch size for processing
            
        Returns:
            numpy.ndarray: Document embeddings
        """
        # Process texts in batches
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use the [CLS] token embedding as the document embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        return np.vstack(embeddings)
    
    def save(self, path):
        """
        Save the transformer model and tokenizer to a directory.
        
        Args:
            path (str): Path to save the model and tokenizer
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def load(cls, path):
        """
        Load a transformer model and tokenizer from a directory.
        
        Args:
            path (str): Path to the saved model and tokenizer
            
        Returns:
            TransformerFeatureExtractor: Loaded feature extractor
        """
        extractor = cls(model_name=path)
        return extractor


class FeatureCombiner:
    """
    Combine multiple feature extractors.
    """
    
    def __init__(self, extractors):
        """
        Initialize the feature combiner.
        
        Args:
            extractors (list): List of feature extractors
        """
        self.extractors = extractors
    
    def fit_transform(self, texts):
        """
        Fit all extractors and transform the texts.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Combined features
        """
        features = []
        
        for extractor in self.extractors:
            if hasattr(extractor, 'fit_transform'):
                feature = extractor.fit_transform(texts)
            else:
                extractor.fit(texts)
                feature = extractor.transform(texts)
            
            # Convert sparse matrix to dense if needed
            if hasattr(feature, 'toarray'):
                feature = feature.toarray()
            
            features.append(feature)
        
        # Concatenate features
        return np.hstack(features)
    
    def transform(self, texts):
        """
        Transform the texts using all extractors.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Combined features
        """
        features = []
        
        for extractor in self.extractors:
            feature = extractor.transform(texts)
            
            # Convert sparse matrix to dense if needed
            if hasattr(feature, 'toarray'):
                feature = feature.toarray()
            
            features.append(feature)
        
        # Concatenate features
        return np.hstack(features)


def extract_features(df, text_column='cleaned_text', title_column='cleaned_title', 
                    method='tfidf', **kwargs):
    """
    Extract features from a DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame
        text_column (str): The name of the text column
        title_column (str): The name of the title column
        method (str): The feature extraction method ('tfidf', 'word2vec', 'transformer')
        **kwargs: Additional arguments for the feature extractor
        
    Returns:
        tuple: (features, feature_extractor)
    """
    # Combine text and title if both are available
    if text_column in df.columns and title_column in df.columns:
        texts = df[text_column].fillna('') + ' ' + df[title_column].fillna('')
    elif text_column in df.columns:
        texts = df[text_column].fillna('')
    elif title_column in df.columns:
        texts = df[title_column].fillna('')
    else:
        raise ValueError(f"Neither {text_column} nor {title_column} found in DataFrame")
    
    # Extract features based on the specified method
    if method == 'tfidf':
        extractor = TfidfFeatureExtractor(**kwargs)
        features = extractor.fit_transform(texts)
    elif method == 'word2vec':
        extractor = Word2VecFeatureExtractor(**kwargs)
        features = extractor.fit_transform(texts)
    elif method == 'transformer':
        extractor = TransformerFeatureExtractor(**kwargs)
        features = extractor.transform(texts)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return features, extractor


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract features from a dataset')
    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output features')
    parser.add_argument('--text-column', type=str, default='cleaned_text', help='Name of the text column')
    parser.add_argument('--title-column', type=str, default='cleaned_title', help='Name of the title column')
    parser.add_argument('--method', type=str, default='tfidf', choices=['tfidf', 'word2vec', 'transformer'],
                        help='Feature extraction method')
    parser.add_argument('--model-output', type=str, help='Path to save the feature extractor')
    
    args = parser.parse_args()
    
    # Load the dataset
    df = pd.read_csv(args.input)
    
    # Extract features
    features, extractor = extract_features(df, args.text_column, args.title_column, args.method)
    
    # Save features
    if hasattr(features, 'toarray'):
        np.save(args.output, features.toarray())
    else:
        np.save(args.output, features)
    
    # Save the feature extractor if specified
    if args.model_output:
        extractor.save(args.model_output)
    
    print(f"Features extracted and saved to {args.output}") 