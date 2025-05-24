import streamlit as st
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_styles import load_css
from shared_components import hide_sidebar_items

# Page config
st.set_page_config(page_title="FAQ", page_icon="❓", layout="wide")

# Load shared CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Add additional CSS to ensure text visibility
st.markdown("""
<style>
/* Ensure all text is visible with proper contrast */
p, span, div, li, a, label, text {
    color: #333333 !important;
}
</style>
""", unsafe_allow_html=True)

# Hide Model Explanation and Prediction from sidebar
hide_sidebar_items()

# Title and introduction
st.title("❓ Frequently Asked Questions")
st.write("Find answers to common questions about the Fake News Detection System.")

# Main content - FAQ items
st.markdown("""
## General Questions
""")

with st.expander("What is the Fake News Detection System?", expanded=True):
    st.markdown("""
    The Fake News Detection System is an AI-powered tool that helps users identify potentially false or misleading news articles. It uses machine learning algorithms, natural language processing, and large language models to analyze text content and provide an assessment of its credibility.

    The system examines various aspects of text including content, writing style, sentiment, and factual claims to make its determination.
    """)

with st.expander("How accurate is the system?", expanded=False):
    st.markdown("""
    The system achieves approximately 85-90% accuracy on benchmark datasets. However, it's important to understand that:

    - No automated system is perfect at detecting fake news
    - The system provides a probability score, not an absolute determination
    - Results should be used as one tool among many for evaluating news credibility
    - Longer, more detailed articles typically yield more accurate results

    We recommend using the system as a starting point for further investigation rather than as a definitive authority.
    """)

with st.expander("What types of fake news can the system detect?", expanded=False):
    st.markdown("""
    The system is trained to detect several types of problematic content:

    - **Fabricated content**: Completely false information
    - **Manipulated content**: Genuine information that has been distorted
    - **Misleading content**: Misleading use of information to frame issues or individuals
    - **False context**: Genuine content shared with false contextual information
    - **Imposter content**: Genuine sources that are impersonated

    It may be less effective at detecting:
    - Subtle forms of bias without factual inaccuracies
    - Satire or parody (though these should ideally be classified as "fake" but with lower confidence)
    - Very short texts with limited information
    """)

st.markdown("""
## Technical Questions
""")

with st.expander("How does the system work?", expanded=False):
    st.markdown("""
    The system works through several integrated components:

    1. **Text Processing**: The input text is cleaned, tokenized, and processed to extract features

    2. **Feature Extraction**: The system analyzes:
       - Content features (words, phrases, topics)
       - Stylometric features (writing style, punctuation, sentence structure)
       - Sentiment features (emotional tone, subjectivity)

    3. **Machine Learning Classification**: A trained model evaluates these features to determine the likelihood of the content being fake

    4. **Explainable AI**: Techniques like LIME and SHAP are used to explain which features influenced the prediction

    5. **LLM Verification**: For additional verification, large language models can be used to fact-check specific claims against known information
    """)

with st.expander("What data was the system trained on?", expanded=False):
    st.markdown("""
    The system was trained on a diverse dataset of news articles that includes:

    - Articles from mainstream news sources
    - Articles from known fake news websites
    - Articles from partisan sources across the political spectrum
    - Articles covering various topics (politics, health, science, entertainment, etc.)

    The training data was carefully balanced to avoid political or topical bias, and was annotated with verified labels (real or fake) based on fact-checking by professional organizations.

    The system is regularly updated with new training data to keep up with evolving patterns of misinformation.
    """)

with st.expander("Do I need special hardware to run the system?", expanded=False):
    st.markdown("""
    The base system runs on standard hardware, but the LLM verification feature has additional requirements:

    **Minimum requirements:**
    - Modern CPU (2+ cores)
    - 8GB RAM
    - 1GB free disk space

    **Recommended for LLM verification:**
    - Modern multi-core CPU
    - 16GB+ RAM
    - GPU with 4GB+ VRAM (for faster LLM processing)
    - 10GB+ free disk space (for LLM models)

    The system can run on most desktop and laptop computers, but performance may vary depending on your hardware.
    """)

st.markdown("""
## Usage Questions
""")

with st.expander("How do I analyze an article?", expanded=False):
    st.markdown("""
    To analyze an article:

    1. Navigate to the "Prediction" page using the sidebar
    2. Copy and paste the full text of the article into the text area
    3. Click the "Analyze Article" button
    4. Review the results, which include:
       - Classification (Real or Fake)
       - Confidence score
       - Word importance visualization
       - Feature importance breakdown

    For best results, include the full article text including the headline.
    """)

with st.expander("How do I use the LLM verification feature?", expanded=False):
    st.markdown("""
    To use the LLM verification feature:

    1. Make sure Ollama is installed and running on your system
    2. Pull a suitable language model (e.g., `ollama pull llama2`)
    3. Navigate to the "LLM Verification" page
    4. Paste the article text you want to verify
    5. Click "Verify with LLM"
    6. Review the verification results, which include:
       - Factual accuracy assessment
       - Reasoning provided by the LLM
       - Internet search results (if enabled)

    Note that LLM verification requires more computational resources and may take longer than basic analysis.
    """)

with st.expander("Can I use the system offline?", expanded=False):
    st.markdown("""
    Yes, the core prediction functionality works offline, but with some limitations:

    - The basic prediction model works completely offline
    - The LLM verification feature requires Ollama to be installed locally, but can work offline with limitations
    - Internet search verification (if enabled) requires an internet connection

    To use the system in a fully offline environment:
    1. Make sure all models are downloaded before going offline
    2. Disable any features that require internet connectivity
    3. Use local LLM models through Ollama
    """)

# Footer
st.markdown("---")
st.markdown("Don't see your question answered here? Contact our support team for more assistance.")
