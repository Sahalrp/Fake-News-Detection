import streamlit as st
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_styles import load_css
from shared_components import hide_sidebar_items

# Page config
st.set_page_config(page_title="Documentation", page_icon="📚", layout="wide")

# Load shared CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Add additional CSS specifically for the Documentation page to hide HTML code
st.markdown("""
<style>
/* Ensure all text is visible with proper contrast */
p, span, div, li, a, label, text {
    color: #333333 !important;
}

/* Additional CSS to hide HTML code in Documentation page */
pre, code, .stCodeBlock, div[data-testid="stCodeBlock"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
}

/* Hide any paragraph or div containing HTML tags */
p:contains("<"), div:contains("<"), p:contains(">"), div:contains(">") {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
}

/* Hide specific patterns seen in the screenshots */
p:contains("display: grid;"), div:contains("display: grid;"),
p:contains("border-radius:"), div:contains("border-radius:"),
p:contains("margin-top:"), div:contains("margin-top:"),
p:contains("color: #"), div:contains("color: #"),
p:contains("background:"), div:contains("background:"),
p:contains("display: flex;"), div:contains("display: flex;"),
p:contains("font-weight:"), div:contains("font-weight:"),
p:contains("font-size:"), div:contains("font-size:"),
p:contains("padding:"), div:contains("padding:"),
p:contains("margin-bottom:"), div:contains("margin-bottom:"),
p:contains("gap:"), div:contains("gap:"),
p:contains("background-color:"), div:contains("background-color:") {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
    opacity: 0 !important;
}

/* Force expander content to be visible */
.streamlit-expanderContent {
    display: block !important;
    visibility: visible !important;
    height: auto !important;
    opacity: 1 !important;
    overflow: visible !important;
}

/* Make sure the expander headers are visible */
.streamlit-expanderHeader {
    display: block !important;
    visibility: visible !important;
    height: auto !important;
    opacity: 1 !important;
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)

# Hide Model Explanation and Prediction from sidebar
hide_sidebar_items()

# Page title with animation
st.markdown("""
<div class="page-title">
    <h1>📚 Documentation</h1>
    <p>Welcome to the Fake News Detection System documentation. Here you'll find detailed information about how to use the system and understand its features.</p>
</div>
""", unsafe_allow_html=True)

# Introduction card
st.markdown("""
<div class="slide-in" style="background: linear-gradient(135deg, rgba(58, 80, 107, 0.05), rgba(91, 192, 190, 0.05));
            padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
    <div style="display: flex; align-items: flex-start; gap: 1rem;">
        <div style="font-size: 2.5rem; color: #5BC0BE; margin-top: 0.5rem;">📖</div>
        <div>
            <h3 style="margin-top: 0; margin-bottom: 0.5rem; color: #3A506B;">Getting Started</h3>
            <p style="color: #6c757d; margin-bottom: 0;">
                This documentation provides comprehensive information about the Fake News Detection System,
                including how to use its features, understand the results, and get the most out of the system.
                Use the sections below to navigate to specific topics.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content with improved styling using Streamlit components
st.header("🔍 System Overview")

# System overview using Streamlit components
st.write("The Fake News Detection System is designed to help users identify potentially false news articles using machine learning and natural language processing techniques. The system analyzes various aspects of text including:")

# Create a grid layout with columns
col1, col2 = st.columns(2)

with col1:
    # Content Analysis card
    st.info("### 📊 Content Analysis\n\nExamines the actual text content for patterns associated with fake news")

    # Sentiment Analysis card
    st.info("### 😊 Sentiment Analysis\n\nEvaluates emotional tone, which may indicate bias or manipulation")

with col2:
    # Stylometric Analysis card
    st.info("### ✍️ Stylometric Analysis\n\nAnalyzes writing style, which can differ between real and fake news")

    # LLM Verification card
    st.info("### 🤖 LLM Verification\n\nUses large language models to fact-check claims against known information")

# Features section with improved styling using Streamlit components
st.header("✨ Key Features")
st.write("Explore the main features of the Fake News Detection System:")

# Create expandable sections for each feature with improved styling
with st.expander("📊 Text Analysis", expanded=False):
    # Use Streamlit components instead of HTML
    st.subheader("📊 Text Analysis")

    st.write("The Text Analysis module provides statistical insights about the dataset and text patterns:")

    # Create a grid layout with columns
    col1, col2 = st.columns(2)

    with col1:
        st.info("**Distribution Analysis**\n\nReal vs. fake news distribution in the dataset")
        st.info("**Sentiment Distribution**\n\nEmotional tone and subjectivity analysis")

    with col2:
        st.info("**Text Length Analysis**\n\nCharacter and word count patterns")
        st.info("**Stylometric Patterns**\n\nWriting style characteristics and punctuation usage")

    st.caption("This helps users understand the characteristics that differentiate real from fake news.")

# Model Explanation and Prediction sections are hidden but code is preserved
if False:
    with st.expander("🔍 Model Explanation", expanded=False):
        # Use Streamlit components instead of HTML
        st.subheader("🔍 Model Explanation")

        st.write("The Model Explanation module makes the AI decision-making process transparent:")

        # Create a grid layout with columns
        col1, col2 = st.columns(2)

        with col1:
            st.info("**Feature Importance**\n\nVisualizes which words and features have the most influence on predictions")
            st.info("**SHAP Values**\n\nSHapley Additive exPlanations showing feature contributions")

        with col2:
            st.info("**LIME Explanations**\n\nLocal Interpretable Model-agnostic Explanations for specific predictions")
            st.info("**Word Impact Analysis**\n\nHighlights which words push toward real or fake classifications")

        st.caption("These techniques help users understand why the model classified an article as real or fake.")

if False:
    with st.expander("🎯 Prediction", expanded=False):
        # Use Streamlit components instead of HTML
        st.subheader("🎯 Prediction")

        st.write("The Prediction module allows users to analyze specific news articles:")

        # Create a numbered list using Streamlit markdown
        st.markdown("1. **Paste or upload news text**")
        st.markdown("2. **Get real-time classification results**")
        st.markdown("3. **View confidence scores**")
        st.markdown("4. **See which parts of the text influenced the prediction**")
        st.markdown("5. **Understand the reasoning behind the classification**")

        # Add a note with Streamlit's native components
        st.info("For best results, provide complete news articles with headlines and multiple paragraphs.")

with st.expander("🤖 LLM Verification", expanded=False):
    # Use Streamlit components instead of HTML
    st.subheader("🤖 LLM Verification")

    st.write("The LLM Verification module uses large language models to fact-check claims:")

    # Create a grid layout with columns
    col1, col2 = st.columns(2)

    with col1:
        st.info("**Claim Extraction**\n\nAutomatically identifies key factual claims from articles")
        st.info("**Fact Comparison**\n\nCompares claims against known facts from search results")

    with col2:
        st.info("**Web Search**\n\nSearches the internet for verification of each claim")
        st.info("**Reasoning & Verdict**\n\nProvides detailed reasoning and final verification verdict")

    # Add a note with Streamlit's native components
    st.warning("This feature requires Ollama to be installed and running on your system.")

# Technical details with improved styling using Streamlit components
st.header("⚙️ Technical Details")

st.write("The Fake News Detection System uses several machine learning and natural language processing techniques:")

# Create a grid layout with columns for technical details
col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 🧠 Model\n\nLogistic Regression with TF-IDF vectorization for efficient text classification")
    st.info("### 🤖 LLM Integration\n\nUses Ollama for local LLM inference with privacy and control")

with col2:
    st.info("### 📊 Features\n\nText content, stylometric features, and sentiment analysis")
    st.info("### 🌐 Web Search\n\nIntegrates with search APIs for real-time fact verification")

with col3:
    st.info("### 🔍 Explainability\n\nLIME and SHAP for model interpretability and transparent predictions")
    st.info("### 🛠️ Tech Stack\n\nBuilt with Python using scikit-learn, NLTK, spaCy, Streamlit, and Plotly")

# System Requirements section
st.subheader("System Requirements")

# Create a grid layout with columns for system requirements
req_col1, req_col2 = st.columns(2)

with req_col1:
    st.success("### 💻 Operating System\n\nWindows, macOS, or Linux")
    st.success("### 🧠 Memory\n\n8GB RAM minimum (16GB recommended)")

with req_col2:
    st.success("### 🧮 Processor\n\nModern multi-core CPU")
    st.success("### 🌐 Internet\n\nRequired for web search verification")

# Usage instructions with improved styling using Streamlit components
st.header("📝 Usage Instructions")

# Getting Started section
st.subheader("🚀 Getting Started")
st.write("Navigate through the system using the sidebar menu on the left. Each module provides different functionality for analyzing and verifying news articles.")

# Using LLM Verification section
st.subheader("🤖 Using LLM Verification")

st.markdown("1. **Install Ollama** if not already installed")
st.code("# Download from ollama.ai/download")

st.markdown("2. **Pull a model** using the terminal")
st.code("ollama pull mistral")

st.markdown("3. **Select the LLM Verification tab** from the sidebar")
st.markdown("4. **Paste the news article** to verify")
st.markdown("5. **Click the \"Verify with LLM\"** button")
st.markdown("6. **Review the verification results**, including:")
st.markdown("   - Verdict (Real, Fake, or Uncertain)")
st.markdown("   - Key claims extracted")
st.markdown("   - Search results and evidence")
st.markdown("   - Reasoning behind the verdict")

# Tips for Best Results
st.subheader("Tips for Best Results")
st.info("""
- Use complete news articles with headlines and multiple paragraphs
- Ensure you have a stable internet connection for web search verification
- The LLM needs internet access to verify facts against online sources
- Always critically evaluate the results and use them as one tool among many for fact-checking
""")

# Add footer with Streamlit components
st.markdown("---")
st.caption("Fake News Detection System Documentation | Version 1.0")

# Add JavaScript to hide HTML code that might be displayed
st.markdown("""
<script>
// JavaScript to hide HTML code that might be displayed
function hideHtmlCode() {
    // Find all elements that might contain code
    const elements = document.querySelectorAll('p, pre, code, div.stCodeBlock, div[data-testid="stCodeBlock"]');

    // Code patterns to look for
    const codePatterns = [
        '<div', '<button', '<span', '<a', '<p', '</', 'style=', 'class=', 'onclick=', 'href=',
        'function', 'def ', 'import ', 'return ', 'const ', 'var ', 'let ', 'document.',
        '@keyframes', '<script', '<style', '<iframe', '<html', '<body', '<head', '<meta',
        'display:', 'margin:', 'padding:', 'color:', 'background-color:'
    ];

    elements.forEach(el => {
        // Get text content
        const text = el.innerText || el.textContent;

        // Skip empty elements
        if (!text) return;

        // Check if element contains code patterns
        const containsCode = codePatterns.some(pattern => text.includes(pattern));

        if (containsCode) {
            // Find the closest container to hide
            const container = el.closest('.element-container') || el;
            container.style.display = 'none';
            container.style.visibility = 'hidden';
            container.style.height = '0';
            container.style.overflow = 'hidden';
            container.style.margin = '0';
            container.style.padding = '0';
            container.style.opacity = '0';

            // If it's a code block, hide it directly
            if (el.classList.contains('stCodeBlock') ||
                (el.hasAttribute('data-testid') && el.getAttribute('data-testid') === 'stCodeBlock')) {
                el.style.display = 'none';
                el.style.visibility = 'hidden';
                el.style.height = '0';
                el.style.overflow = 'hidden';
                el.style.margin = '0';
                el.style.padding = '0';
                el.style.opacity = '0';
            }
        }
    });

    // Specifically target code blocks
    const codeBlocks = document.querySelectorAll('.stCodeBlock, [data-testid="stCodeBlock"]');
    codeBlocks.forEach(block => {
        block.style.display = 'none';
        block.style.visibility = 'hidden';
        block.style.height = '0';
        block.style.overflow = 'hidden';
        block.style.margin = '0';
        block.style.padding = '0';
        block.style.opacity = '0';
    });
}

// Run when the page loads
document.addEventListener('DOMContentLoaded', function() {
    // Initial run
    hideHtmlCode();

    // Run again after a short delay to catch elements that might load later
    setTimeout(hideHtmlCode, 500);
    setTimeout(hideHtmlCode, 1000);
    setTimeout(hideHtmlCode, 2000);
});

// Use MutationObserver to detect DOM changes
const observer = new MutationObserver(function(mutations) {
    // Run our function when DOM changes
    hideHtmlCode();
});

// Start observing with a comprehensive configuration
observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    characterData: true
});
</script>
""", unsafe_allow_html=True)

