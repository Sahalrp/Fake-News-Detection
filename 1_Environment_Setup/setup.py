from setuptools import setup, find_packages

setup(
    name="fake_news_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "nltk>=3.8.1",
        "spacy>=3.5.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning system for detecting fake news",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)










