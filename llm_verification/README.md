# LLM-based News Verification Module

This module adds LLM-based verification capabilities to the Fake News Detection system. It uses a locally running Large Language Model (LLM) through Ollama, combined with web search functionality to verify news articles.

## Features

- **Local LLM Processing**: Uses Ollama to run LLMs locally on your machine
- **Web Search Integration**: Searches the internet for information to verify news claims
- **Claim Extraction**: Automatically identifies key claims in news articles
- **Detailed Analysis**: Provides in-depth analysis of each claim with evidence
- **Privacy-Focused**: All processing happens locally, no data sent to external AI services

## Architecture

The module consists of three main components:

1. **OllamaClient**: Interface to the locally running Ollama service
2. **WebSearch**: Handles web searches and content extraction
3. **NewsVerifier**: Combines the LLM and web search to verify news articles

## Requirements

- macOS (Apple Silicon M1/M2/M3 recommended)
- Python 3.8+
- 16GB RAM recommended
- Ollama application installed

## Installation

See the [INSTALL.md](INSTALL.md) file for detailed installation instructions.

## Usage

### From Python

```python
from llm_verification import NewsVerifier

# Initialize the verifier
verifier = NewsVerifier(model_name="mistral", temperature=0.3)

# Verify an article
result = verifier.verify_article("Your news article text here...")

# Access the results
print(f"Verdict: {result['verdict']}")
print(f"Confidence: {result['confidence']}")
print(f"Analysis: {result['analysis']}")
```

### From Streamlit

The module integrates with the Streamlit app through the "LLM Verification" page. Users can:

1. Enter news text
2. Click "Verify with LLM"
3. View the verification results, including:
   - Verdict (Real/Fake)
   - Confidence level
   - Key claims identified
   - Analysis of each claim
   - Web search results

## Supported Models

The following models are recommended for use with this module:

- **Mistral 7B**: Good balance of performance and resource usage
- **Llama2 7B**: Alternative option with good performance
- **Phi-2**: Smaller model that works well on limited hardware

## Limitations

- Verification quality depends on the LLM model used
- Web search results may be limited or outdated
- Processing time depends on hardware capabilities
- The LLM may occasionally hallucinate or make errors
- Internet connection required for web search functionality

## Future Improvements

- Add support for more search engines
- Implement caching for faster repeated verifications
- Add source credibility assessment
- Improve claim extraction accuracy
- Add support for image analysis in news articles
