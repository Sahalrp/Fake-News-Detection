import streamlit as st
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_styles import load_css
from shared_components import hide_sidebar_items

# Page config
st.set_page_config(page_title="Tutorials", page_icon="🎓", layout="wide")

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

# Title and introduction with improved styling
st.markdown("""
<div class="page-title">
    <h1>🎓 Tutorials</h1>
    <p>Learn how to use the Fake News Detection System with these step-by-step tutorials.</p>
</div>
""", unsafe_allow_html=True)

# Main content with improved styling
st.markdown("""
<h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 2rem; margin-bottom: 1.5rem;">
    <span style="color: #5BC0BE;">🚀</span> Getting Started
</h2>
<p style="color: #6c757d; margin-bottom: 2rem;">
    Follow these step-by-step tutorials to learn how to use the Fake News Detection System effectively.
    Each tutorial focuses on a different aspect of the system.
</p>
""", unsafe_allow_html=True)

# Define HTML content as variables to avoid displaying raw HTML

# Tutorial 1 with improved styling
with st.expander("📊 Tutorial 1: Basic News Analysis", expanded=True):
    # Use Streamlit components instead of raw HTML
    st.subheader("📊 Basic News Analysis")
    st.write("This tutorial will guide you through analyzing your first news article.")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">1</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Navigate to the Prediction Page")
        st.write("Click on 'Prediction' in the sidebar or use the Quick Links section to navigate to the prediction page.")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">2</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Input News Text")
        st.write("• Copy a news article from any source")
        st.write("• Paste it into the text area on the Prediction page")
        st.write("• Make sure to include the headline and several paragraphs for best results")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">3</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Analyze the Article")
        st.write("• Click the 'Analyze Article' button")
        st.write("• Wait for the system to process the text")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">4</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Interpret the Results")
        st.write("• Check the prediction (Real or Fake) and confidence score")
        st.write("• Review the word importance visualization to see which words influenced the prediction")
        st.write("• Examine the feature importance to understand the model's decision-making")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">5</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Try Different Articles")
        st.write("• Try analyzing different types of articles to see how the system responds")
        st.write("• Compare results between known reliable sources and questionable sources")

# Tutorial 2 with improved styling
with st.expander("🤖 Tutorial 2: Using LLM Verification", expanded=False):
    # Use Streamlit components instead of raw HTML
    st.subheader("🤖 Using LLM Verification")
    st.write("This tutorial explains how to use the LLM verification feature to fact-check news articles.")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">1</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Set Up Ollama")
        st.write("Before using LLM verification, make sure Ollama is installed and running:")

        st.markdown("""
        **Installation Steps:**
        1. Download Ollama from [ollama.ai/download](https://ollama.ai/download)
        2. Install and run the application
        3. Pull a model using Terminal
        """)

        st.code("""# Install Ollama (if not already installed)
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama
ollama serve

# Pull a suitable model (in a separate terminal)
ollama pull llama2""")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">2</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Navigate to LLM Verification")
        st.write("Click on 'LLM Verification' in the sidebar or use the Quick Links section.")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">3</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Input News Text")
        st.write("• Paste the news article you want to verify")
        st.write("• Optionally, highlight specific claims you want to check")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">4</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Run Verification")
        st.write("• Select the LLM model to use (if multiple are available)")
        st.write("• Click 'Verify with LLM'")
        st.write("• Wait for the verification process to complete")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">5</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Review Results")
        st.write("• Read the LLM's analysis of the article")
        st.write("• Check the factual accuracy assessment")
        st.write("• Review the reasoning provided by the LLM")
        st.write("• Examine any internet search results that were used for verification")

# Tutorial 3 with improved styling
with st.expander("🔍 Tutorial 3: Understanding Model Explanations", expanded=False):
    # Use Streamlit components instead of raw HTML
    st.subheader("🔍 Understanding Model Explanations")
    st.write("This tutorial helps you interpret the model explanations to better understand why an article was classified as real or fake.")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">1</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Analyze an Article")
        st.write("First, analyze an article using the Prediction page as described in Tutorial 1.")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">2</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Navigate to Model Explanation")
        st.write("After getting a prediction, you can explore more detailed explanations:")
        st.write("• Look at the 'Word/Phrase Impact' section")
        st.write("• Blue bars indicate words that suggest real news")
        st.write("• Red bars indicate words that suggest fake news")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">3</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Interpret LIME Explanations")
        st.write("LIME explanations show how individual words and phrases affected the prediction:")
        st.write("• Longer bars indicate stronger influence")
        st.write("• The direction (positive or negative) shows whether it pushed toward real or fake")
        st.write("• Focus on the top influencing factors")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">4</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Understand SHAP Values")
        st.write("SHAP values provide another perspective on feature importance:")
        st.write("• They show how each feature contributed to pushing the prediction away from the baseline")
        st.write("• They can help identify subtle patterns that influenced the model")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">5</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Compare Global vs. Local Importance")
        st.write("• Global feature importance shows which features are generally important across all predictions")
        st.write("• Local importance (LIME and SHAP) shows what was important for this specific prediction")
        st.write("• Comparing these can help you understand if this article follows typical patterns")

# Tutorial 4 with improved styling
with st.expander("📝 Tutorial 4: Analyzing Text Patterns", expanded=False):
    # Use Streamlit components instead of raw HTML
    st.subheader("📝 Analyzing Text Patterns")
    st.write("This tutorial shows how to use the Text Analysis page to understand patterns in news articles.")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">1</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Navigate to Text Analysis")
        st.write("Click on 'Text Analysis' in the sidebar or use the Quick Links section.")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">2</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Explore Dataset Statistics")
        st.write("• Review the distribution of real vs. fake news")
        st.write("• Check the basic statistics about article length, word count, etc.")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">3</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Analyze Sentiment Patterns")
        st.write("• Examine the sentiment polarity distribution")
        st.write("• Compare sentiment patterns between real and fake news")
        st.write("• Look for differences in subjectivity scores")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">4</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Study Stylometric Features")
        st.write("• Review punctuation usage patterns")
        st.write("• Analyze sentence length distributions")
        st.write("• Look at quote usage differences")

    col1, col2 = st.columns([1, 20])
    with col1:
        st.markdown('<div style="background-color: #5BC0BE; color: white; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.9rem;">5</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("#### Apply Insights to Your Analysis")
        st.write("• Use the patterns you've identified to better understand prediction results")
        st.write("• Look for these patterns in articles you're analyzing")
        st.write("• Consider how these patterns might relate to credibility")

# Footer with improved styling
st.markdown("""
<div style="margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #e0e0e0; text-align: center;">
    <p style="color: #6c757d; font-size: 0.9rem;">
        Need more help? Check the <a href="/Documentation" style="color: #5BC0BE; text-decoration: none;">Documentation</a> page or contact our support team.
    </p>
</div>
""", unsafe_allow_html=True)
