# Fake News Detection System with Explainable AI

## Overview
This project implements an advanced Fake News Detection System using state-of-the-art Natural Language Processing (NLP) and Machine Learning (ML) techniques, enhanced with Explainable AI (XAI) capabilities. The system aims to classify news articles as real or fake while providing transparent explanations for its predictions.

## Key Features
- Multiple ML models including traditional (Logistic Regression, Random Forest) and transformer-based (BERT, RoBERTa, DistilBERT) architectures
- Advanced text feature extraction using TF-IDF, Word2Vec, and GloVe embeddings
- Explainable AI integration with SHAP and LIME for model interpretability
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-score)
- Interactive web interface for real-time news classification
- Support for multilingual news content
- Extensive testing and validation framework

## Project Structure
```
.
├── app/                    # Web application and UI components
├── data/                   # Dataset storage and preprocessing scripts
├── notebooks/             # Jupyter notebooks for analysis and experimentation
├── src/                   # Source code for the core ML pipeline
│   ├── models/           # ML model implementations
│   ├── features/         # Feature extraction and processing
│   ├── explainer/        # XAI implementation (SHAP, LIME)
│   └── utils/            # Helper functions and utilities
├── tests/                 # Unit tests and integration tests
├── requirements.txt       # Project dependencies
└── setup.py              # Package configuration
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Data Preparation:
```bash
python src/data/prepare_dataset.py
```

2. Model Training:
```bash
python src/models/train.py --model [bert|roberta|lr|rf]
```

3. Run Web Interface:
```bash
streamlit run app/main.py
```

## Model Performance
The system employs multiple models and evaluates them using various metrics:

| Model      | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|---------|-----------|
| BERT       | 0.94     | 0.93      | 0.95    | 0.94      |
| RoBERTa    | 0.95     | 0.94      | 0.96    | 0.95      |
| LogReg     | 0.89     | 0.88      | 0.90    | 0.89      |
| RandomForest| 0.91     | 0.90      | 0.92    | 0.91      |

## Explainability
The system provides two types of explanations:
1. SHAP (SHapley Additive exPlanations) values for feature importance
2. LIME (Local Interpretable Model-agnostic Explanations) for local predictions

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this project in your research, please cite:
```
@software{fake_news_detection,
  title = {Fake News Detection System with Explainable AI},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/fake-news-detection}
}
```

## Contact
For questions and feedback, please open an issue or contact [your-email@example.com].
