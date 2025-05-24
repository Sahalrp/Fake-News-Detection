import re
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Union, Optional

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class TextPreprocessor:
    """Class for preprocessing text data."""
    
    def __init__(self,
                 remove_stopwords: bool = True,
                 lemmatize: bool = True,
                 language: str = 'english'):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
            language: Language for stopwords (default: 'english')
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.language = language
        
        if remove_stopwords:
            self.stopwords = set(stopwords.words(language))
        
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean the input text by removing special characters and normalizing whitespace.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from the token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        return [token for token in tokens if token not in self.stopwords]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to the tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: Union[str, List[str]],
                  return_tokens: bool = False) -> Union[str, List[str]]:
        """
        Apply full preprocessing pipeline to the input text.
        
        Args:
            text: Input text or list of texts to preprocess
            return_tokens: Whether to return tokens instead of joined text
            
        Returns:
            Preprocessed text or list of texts
        """
        if isinstance(text, list):
            return [self.preprocess(t, return_tokens) for t in text]
        
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Apply lemmatization if enabled
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        if return_tokens:
            return tokens
        
        return ' '.join(tokens)
    
    def get_vocabulary(self, texts: List[str]) -> set:
        """
        Get the vocabulary from a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Set of unique tokens in the texts
        """
        tokens = []
        for text in texts:
            tokens.extend(self.preprocess(text, return_tokens=True))
        return set(tokens) 