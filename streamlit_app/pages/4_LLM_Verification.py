"""
LLM-based news verification page for the Streamlit app.
"""
import os
import sys
import streamlit as st
import pandas as pd
import time
import re
import html
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob

# Add the parent directory to the path so we can import the llm_verification module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

# Add parent directory to path to import utils and shared styles
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_styles import load_css
from shared_components import hide_sidebar_items
from utils import extract_sentiment_features, extract_stylometric_features

# Import the LLM verification module
try:
    from llm_verification import NewsVerifier, OllamaClient
except ImportError:
    st.error("Could not import the LLM verification module. Please make sure it's installed.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="LLM News Verification",
    page_icon="🔍",
    layout="wide"
)

# Load shared CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Hide Model Explanation and Prediction from sidebar
hide_sidebar_items()

# Additional custom CSS for this page
st.markdown("""
<style>
    /* Ensure all text is visible with proper contrast */
    p, span, div, li, a, label, text {
        color: #333333 !important;
    }
    
    /* Verification card styling */
    .verification-card {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .verification-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }

    .verification-card h2 {
        margin-bottom: 1rem !important;
        font-weight: 700 !important;
        color: #1C2541 !important;
    }

    .real {
        background: linear-gradient(to bottom right, rgba(46, 125, 50, 0.1), white);
        border-left: 5px solid #2E7D32;
    }

    .fake {
        background: linear-gradient(to bottom right, rgba(198, 40, 40, 0.1), white);
        border-left: 5px solid #C62828;
    }

    .unknown {
        background: linear-gradient(to bottom right, rgba(255, 209, 102, 0.1), white);
        border-left: 5px solid #FFD166;
    }

    /* Confidence meter styling */
    .confidence-meter {
        height: 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
        overflow: hidden;
    }

    .fill {
        height: 100%;
        border-radius: 5px;
        transition: width 1s ease-in-out;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }

    /* Claim box styling */
    .claim-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 3px solid #5BC0BE;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    /* Search result styling */
    .search-result {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 3px solid #5BC0BE;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    /* Model selection card */
    .model-selection-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
    }

    /* Status indicator styling */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-indicator.success {
        background-color: #2E7D32;
        box-shadow: 0 0 8px rgba(46, 125, 50, 0.5);
    }

    .status-indicator.error {
        background-color: #C62828;
        box-shadow: 0 0 8px rgba(198, 40, 40, 0.5);
    }

    .status-indicator.warning {
        background-color: #FFD166;
        box-shadow: 0 0 8px rgba(255, 209, 102, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# Page title with animation
st.markdown("""
<div class="page-title">
    <h1>🔍 LLM News Verification</h1>
    <p>Verify news articles using a Large Language Model with internet search capabilities</p>
</div>
""", unsafe_allow_html=True)

# Introduction card
st.markdown("""
<div class="slide-in" style="background: linear-gradient(135deg, rgba(58, 80, 107, 0.05), rgba(91, 192, 190, 0.05));
            padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
    <div style="display: flex; align-items: flex-start; gap: 1rem;">
        <div style="font-size: 2.5rem; color: #5BC0BE; margin-top: 0.5rem;">💡</div>
        <div>
            <h3 style="margin-top: 0; margin-bottom: 0.5rem; color: #3A506B;">How It Works</h3>
            <p style="color: #6c757d; margin-bottom: 0.5rem;">
                This feature uses a locally running Large Language Model to verify news articles by:
            </p>
            <ol style="color: #6c757d; margin-bottom: 0.5rem;">
                <li>Extracting key factual claims from the article</li>
                <li>Searching the web to find supporting or contradicting information</li>
                <li>Analyzing the search results to determine if the article appears to be real or fake</li>
            </ol>
            <div style="background-color: rgba(91, 192, 190, 0.1); padding: 0.75rem; border-radius: 4px; margin-top: 0.75rem;
                        border-left: 3px solid #5BC0BE;">
                <p style="color: #3A506B; margin: 0; font-weight: 500;">
                    <span style="color: #5BC0BE; font-weight: bold;">NEW:</span> You can now compare results between different LLM models (DeepSeek and Mistral) to see how they evaluate the same content. Enable this feature in the sidebar settings.
                </p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with improved styling to match other pages
with st.sidebar:
    # Logo and title
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=60)
    with col2:
        st.markdown("<h2 style='margin-top:0.5rem; color: white;'>LLM<br>Verification</h2>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Check if Ollama is available
    ollama_client = OllamaClient()
    ollama_available = ollama_client.check_availability()

    # Ollama status with custom indicator
    st.markdown("### 🤖 Ollama Status")

    if ollama_available:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem; background-color: rgba(91, 192, 190, 0.2);
                    padding: 0.5rem; border-radius: 4px;">
            <div class="status-indicator success"></div>
            <div style="font-weight: 500;">Ollama is running</div>
        </div>
        """, unsafe_allow_html=True)

        # List available models
        models = ollama_client.list_models()
        model_names = [model["name"] for model in models] if models else []

        if model_names:
            st.markdown(f"""
            <div style="background-color: rgba(255,255,255,0.15); padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem;
                        border-left: 3px solid #5BC0BE;">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">Available models:</div>
                <div style="font-size: 0.9rem; opacity: 0.9; font-family: monospace;">{", ".join(model_names)}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem; background-color: rgba(255, 209, 102, 0.2);
                        padding: 0.5rem; border-radius: 4px;">
                <div class="status-indicator warning"></div>
                <div style="font-weight: 500;">No models found. Please pull a model.</div>
            </div>

            <div style="background-color: rgba(255,255,255,0.15); padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem;
                        font-family: monospace; font-size: 0.85rem; border-left: 3px solid #FFD166;">
                ollama pull deepseek-r1:7b<br>
                # or<br>
                ollama pull mistral
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem; background-color: rgba(198, 40, 40, 0.2);
                    padding: 0.5rem; border-radius: 4px;">
            <div class="status-indicator error"></div>
            <div style="font-weight: 500;">Ollama not running</div>
        </div>

        <div style="background-color: rgba(255,255,255,0.15); padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem;
                    font-size: 0.9rem; border-left: 3px solid #C62828;">
            Please install and start Ollama:
            <ol style="margin-top: 0.5rem; margin-bottom: 0; padding-left: 1.5rem;">
                <li>Download from <a href="https://ollama.ai/download" target="_blank" style="color: white;">ollama.ai/download</a></li>
                <li>Install and run the application</li>
                <li>Refresh this page</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # Model settings section
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Model Settings")

    # Model selection with improved styling
    model_name = st.selectbox(
        "Select LLM Model",
        ["deepseek-r1:7b", "mistral", "llama2", "phi"],
        index=0,
        help="Choose which language model to use for verification"
    )

    # Add model comparison option
    compare_models = st.checkbox(
        "Compare with another model",
        value=False,
        help="Run verification with both DeepSeek and Mistral models and compare results"
    )

    # If comparison is enabled, show the second model
    if compare_models:
        # Determine the comparison model (if primary is deepseek, use mistral; otherwise use deepseek)
        if model_name == "deepseek-r1:7b":
            comparison_model = "mistral"
        else:
            comparison_model = "deepseek-r1:7b"

        st.markdown(f"""
        <div style="background-color: rgba(91, 192, 190, 0.1); padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem;
                    border-left: 3px solid #5BC0BE;">
            <div style="font-weight: 500; margin-bottom: 0.25rem;">Comparison Model:</div>
            <div style="font-size: 0.9rem; opacity: 0.9; font-family: monospace;">{comparison_model}</div>
        </div>
        """, unsafe_allow_html=True)

    # Temperature slider with improved styling
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower values make the output more deterministic, higher values more creative"
    )

    # Number of search results with improved styling
    num_search_results = st.slider(
        "Search Results",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Number of web search results to use for verification"
    )

    # Add Fast Mode toggle
    fast_mode = st.checkbox(
        "Fast Mode (Recommended)",
        value=True,
        help="Enable faster verification with fewer claims and searches. Recommended for most users."
    )

    # Save fast mode setting to session state for use in model comparison
    st.session_state['fast_mode'] = fast_mode

    # Display model comparison results if available
    if 'comparison_results' in st.session_state:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📊 Model Comparison")

        # Get comparison results
        comp = st.session_state.comparison_results
        primary_model = comp["primary_model"]
        comparison_model = comp["comparison_model"]
        primary_real = comp["primary_real"]
        comparison_real = comp["comparison_real"]
        real_difference = comp["real_difference"]

        # Format model names for display
        primary_display = primary_model.split(":")[0] if ":" in primary_model else primary_model
        comparison_display = comparison_model.split(":")[0] if ":" in comparison_model else comparison_model

        # Determine which model thinks it's more real
        more_real_model = primary_display if primary_real > comparison_real else comparison_display

        # Create a color scale based on the difference
        if real_difference < 10:
            diff_color = "#4ade80"  # Green - models agree
            agreement_text = "Models largely agree"
            agreement_icon = "✓"
        elif real_difference < 20:
            diff_color = "#fde047"  # Yellow - minor disagreement
            agreement_text = "Minor disagreement"
            agreement_icon = "⚠️"
        else:
            diff_color = "#f87171"  # Red - significant disagreement
            agreement_text = "Models disagree"
            agreement_icon = "❗"

        # Display comparison card with proper formatting
        st.markdown(f"""
        <div style="background-color: #1e293b; color: white; padding: 1rem; border-radius: 8px; margin-top: 1rem; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.75rem;">
                <div style="font-size: 1.2rem;">{agreement_icon}</div>
                <div style="font-weight: 600; font-size: 0.9rem;">{agreement_text}</div>
            </div>

            <div style="margin-bottom: 0.75rem;">
                <div style="font-size: 0.8rem; opacity: 0.8; margin-bottom: 0.25rem;">Difference: {real_difference}%</div>
                <div style="height: 6px; background-color: #334155; border-radius: 3px; overflow: hidden;">
                    <div style="width: {min(100, real_difference*2)}%; height: 100%; background-color: {diff_color};"></div>
                </div>
            </div>

            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.5rem;">
                <div style="font-weight: 500;">{primary_display}</div>
                <div style="font-weight: 500;">{comparison_display}</div>
            </div>

            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <div style="font-size: 0.8rem; color: #4ade80;">Real: {primary_real}%</div>
                <div style="font-size: 0.8rem; color: #4ade80;">Real: {comparison_real}%</div>
            </div>

            <div style="display: flex; justify-content: space-between; margin-bottom: 0.75rem;">
                <div style="font-size: 0.8rem; color: #f87171;">Fake: {100-primary_real}%</div>
                <div style="font-size: 0.8rem; color: #f87171;">Fake: {100-comparison_real}%</div>
            </div>

            <div style="font-size: 0.8rem; opacity: 0.9; background-color: #0f172a; padding: 0.5rem; border-radius: 4px;">
                {more_real_model} considers this content more credible
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Add navigation section
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📊 Navigation")

    # Navigation buttons with improved styling
    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.75rem;">
        <a href="/" target="_self" style="text-decoration: none;">
            <div style="background-color: rgba(91, 192, 190, 0.2); padding: 0.75rem; border-radius: 6px; text-align: center;
                        transition: transform 0.2s ease, background-color 0.2s ease; border: 1px solid rgba(91, 192, 190, 0.3);">
                <div style="font-size: 1.5rem;">🏠</div>
                <div style="font-size: 0.9rem; font-weight: 500; margin-top: 0.25rem;">Home</div>
            </div>
        </a>
        <a href="/Prediction" target="_self" style="text-decoration: none;">
            <div style="background-color: rgba(91, 192, 190, 0.2); padding: 0.75rem; border-radius: 6px; text-align: center;
                        transition: transform 0.2s ease, background-color 0.2s ease; border: 1px solid rgba(91, 192, 190, 0.3);">
                <div style="font-size: 1.5rem;">🎯</div>
                <div style="font-size: 0.9rem; font-weight: 500; margin-top: 0.25rem;">Prediction</div>
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

# Main content with improved styling
if not ollama_available:
    # Enhanced warning message
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(198, 40, 40, 0.1), rgba(255, 255, 255, 0.5));
                padding: 1.5rem; border-radius: 10px; margin: 2rem 0; border-left: 4px solid #C62828;">
        <h3 style="margin-top: 0; color: #C62828; display: flex; align-items: center; gap: 0.5rem;">
            <span>⚠️</span> Ollama Not Available
        </h3>
        <p style="color: #3A506B; margin-bottom: 0;">
            Ollama is not running or not installed. Please install and start it to use the LLM verification feature.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced installation instructions
    with st.expander("Installation Instructions", expanded=True):
        st.markdown("""
        <div style="padding: 1rem; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
            <h3 style="color: #3A506B; margin-top: 0;">Installing Ollama</h3>

            <ol style="color: #6c757d; margin-bottom: 0;">
                <li>Download Ollama from <a href="https://ollama.ai/download" target="_blank">ollama.ai/download</a></li>
                <li>Install the application</li>
                <li>Launch Ollama from your Applications folder</li>
                <li>Pull a model by opening Terminal and running:
                    <div style="background-color: #f8f9fa; padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0; font-family: monospace;">
                    ollama pull mistral
                    </div>
                </li>
                <li>Refresh this page</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# Model selection card
st.markdown("""
<div class="model-selection-card">
    <h3 style="margin-top: 0; color: #3A506B; display: flex; align-items: center; gap: 0.5rem;">
        <span style="color: #5BC0BE;">⚙️</span> Verification Settings
    </h3>
    <p style="color: #6c757d; margin-bottom: 0.5rem;">
        Using model: <strong>{model_name}</strong> | Temperature: <strong>{temperature}</strong> | Search results: <strong>{num_search_results}</strong>
    </p>
    <p style="color: #6c757d; margin-bottom: 0; font-size: 0.9rem;">
        <span style="color: #5BC0BE; font-weight: bold;">{fast_mode_status}</span> Fast Mode is {fast_mode_text}. {fast_mode_description}
    </p>
</div>
""".format(
    model_name=model_name,
    temperature=temperature,
    num_search_results=num_search_results,
    fast_mode_status="✓" if fast_mode else "○",
    fast_mode_text="enabled" if fast_mode else "disabled",
    fast_mode_description="Verification will be faster with fewer claims analyzed." if fast_mode else "Verification will be more thorough but slower."
), unsafe_allow_html=True)

# Text input with improved styling
st.markdown("""
<h3 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 2rem; margin-bottom: 1rem;">
    <span style="color: #5BC0BE;">📝</span> Enter News Article
</h3>
""", unsafe_allow_html=True)

# Example section above the text area
with st.expander("See example text", expanded=False):
    st.markdown("""
    <div style="background-color: rgba(91, 192, 190, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <h4 style="margin-top: 0; color: #3A506B;">Example News Article</h4>
        <p style="color: #6c757d; font-style: italic;">
        Scientists have discovered a new species of butterfly in the Amazon rainforest.
        The species, named Morpho azure, displays an unprecedented shade of blue in its wings.
        The research team spent three years studying the butterfly's habitat and behavior.
        Their findings were published in the Journal of Lepidopterology.
        </p>
    </div>
    """, unsafe_allow_html=True)

text_input = st.text_area(
    "Enter news text to verify:",
    height=200,
    help="Paste the news article text you want to verify",
    placeholder="Paste your news article here... (For best results, include the full article with headline and several paragraphs)"
)

# Verification button with improved styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    verify_button = st.button(
        "🔍 Verify with LLM",
        type="primary",
        disabled=not text_input or not ollama_available,
        use_container_width=True
    )

# Function to run verification with multiple models and compare results
def run_model_comparison(text_input, primary_model, comparison_model, temperature):
    """
    Run verification with two different models and compare the results.

    Args:
        text_input: The text to verify
        primary_model: The primary model to use
        comparison_model: The model to compare with
        temperature: Temperature setting for both models

    Returns:
        Dictionary with results from both models and comparison metrics
    """
    # Get fast mode setting from sidebar
    fast_mode = st.session_state.get('fast_mode', True)

    # Initialize verifiers for both models with fast mode
    try:
        # Check if fast_mode is supported by inspecting the __init__ parameters
        import inspect
        init_params = inspect.signature(NewsVerifier.__init__).parameters

        if 'fast_mode' in init_params:
            primary_verifier = NewsVerifier(
                model_name=primary_model,
                temperature=temperature,
                fast_mode=fast_mode  # Use fast mode setting
            )
        else:
            # If fast_mode parameter is not in the signature, don't use it
            primary_verifier = NewsVerifier(
                model_name=primary_model,
                temperature=temperature
            )
    except Exception as e:
        # If there's any error, try without fast_mode
        st.warning(f"Fast mode not supported in this version of NewsVerifier: {str(e)}")
        primary_verifier = NewsVerifier(
            model_name=primary_model,
            temperature=temperature
        )

    try:
        # Use the same approach for the comparison verifier
        if 'fast_mode' in init_params:
            comparison_verifier = NewsVerifier(
                model_name=comparison_model,
                temperature=temperature,
                fast_mode=fast_mode  # Use fast mode setting
            )
        else:
            # If fast_mode parameter is not in the signature, don't use it
            comparison_verifier = NewsVerifier(
                model_name=comparison_model,
                temperature=temperature
            )
    except Exception as e:
        # If there's any error, try without fast_mode
        st.warning(f"Fast mode not supported in this version of NewsVerifier: {str(e)}")
        comparison_verifier = NewsVerifier(
            model_name=comparison_model,
            temperature=temperature
        )

    # Verify with primary model
    primary_result = primary_verifier.verify_article(text_input)

    # Verify with comparison model
    comparison_result = comparison_verifier.verify_article(text_input)

    # Get real/fake percentages
    primary_real = primary_result.get("real_percentage", 50)
    primary_fake = primary_result.get("fake_percentage", 50)
    comparison_real = comparison_result.get("real_percentage", 50)
    comparison_fake = comparison_result.get("fake_percentage", 50)

    # Calculate difference
    real_difference = abs(primary_real - comparison_real)

    # Determine which model thinks it's more real/fake
    more_real_model = primary_model if primary_real > comparison_real else comparison_model
    more_fake_model = primary_model if primary_real < comparison_real else comparison_model

    # Return combined results
    return {
        "primary_result": primary_result,
        "comparison_result": comparison_result,
        "primary_real": primary_real,
        "primary_fake": primary_fake,
        "comparison_real": comparison_real,
        "comparison_fake": comparison_fake,
        "real_difference": real_difference,
        "more_real_model": more_real_model,
        "more_fake_model": more_fake_model,
        "primary_model": primary_model,
        "comparison_model": comparison_model
    }

if verify_button and text_input:
    # Progress bar
    progress_bar = st.progress(0)

    # Step indicators
    st.markdown("""
    <div style="display: flex; justify-content: space-between; margin: 1rem 0;">
        <div style="text-align: center; flex: 1;">
            <div style="background-color: #5BC0BE; color: white; width: 30px; height: 30px; border-radius: 50%;
                        display: flex; align-items: center; justify-content: center; margin: 0 auto;">1</div>
            <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.25rem;">Extracting Claims</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="background-color: #5BC0BE; color: white; width: 30px; height: 30px; border-radius: 50%;
                        display: flex; align-items: center; justify-content: center; margin: 0 auto;">2</div>
            <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.25rem;">Searching Web</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="background-color: #5BC0BE; color: white; width: 30px; height: 30px; border-radius: 50%;
                        display: flex; align-items: center; justify-content: center; margin: 0 auto;">3</div>
            <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.25rem;">Analyzing Results</div>
        </div>
        <div style="text-align: center; flex: 1;">
            <div style="background-color: #5BC0BE; color: white; width: 30px; height: 30px; border-radius: 50%;
                        display: flex; align-items: center; justify-content: center; margin: 0 auto;">4</div>
            <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.25rem;">Text Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Analyzing with LLM and searching the web for verification..."):
        try:
            # Update progress
            progress_bar.progress(20)
            st.markdown("""
            <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 1rem;">
                <span style="color: #5BC0BE;">⏳</span> Extracting key claims from article...
            </div>
            """, unsafe_allow_html=True)

            # Perform sentiment and stylometric analysis
            sentiment_analysis = extract_sentiment_features(text_input)
            stylometric_analysis = extract_stylometric_features(text_input)

            # Store the analyses in session state for later use
            st.session_state['sentiment_analysis'] = sentiment_analysis
            st.session_state['stylometric_analysis'] = stylometric_analysis

            # Simulate step progress
            time.sleep(1)
            progress_bar.progress(40)
            st.markdown("""
            <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 1rem;">
                <span style="color: #5BC0BE;">⏳</span> Searching the web for verification...
            </div>
            """, unsafe_allow_html=True)

            # Check if we should compare models
            if compare_models:
                # Determine comparison model
                if model_name == "deepseek-r1:7b":
                    comparison_model_name = "mistral"
                else:
                    comparison_model_name = "deepseek-r1:7b"

                # Run comparison
                comparison_results = run_model_comparison(
                    text_input,
                    model_name,
                    comparison_model_name,
                    temperature
                )

                # Get the primary result for display
                result = comparison_results["primary_result"]

                # Store comparison results in session state for sidebar display
                st.session_state.comparison_results = comparison_results
            else:
                # Initialize the verifier with single model and fast mode
                try:
                    # Check if fast_mode is supported by inspecting the __init__ parameters
                    import inspect
                    init_params = inspect.signature(NewsVerifier.__init__).parameters

                    if 'fast_mode' in init_params:
                        verifier = NewsVerifier(
                            model_name=model_name,
                            temperature=temperature,
                            fast_mode=fast_mode  # Use the fast mode setting
                        )
                    else:
                        # If fast_mode parameter is not in the signature, don't use it
                        verifier = NewsVerifier(
                            model_name=model_name,
                            temperature=temperature
                        )
                except Exception as e:
                    # If there's any error, try without fast_mode
                    st.warning(f"Fast mode not supported in this version of NewsVerifier: {str(e)}")
                    verifier = NewsVerifier(
                        model_name=model_name,
                        temperature=temperature
                    )

                # Verify the article with single model
                result = verifier.verify_article(text_input)

                # Clear any previous comparison results
                if 'comparison_results' in st.session_state:
                    del st.session_state.comparison_results

            # Update progress
            progress_bar.progress(75)
            st.markdown("""
            <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 1rem;">
                <span style="color: #5BC0BE;">⏳</span> Analyzing search results and generating verdict...
            </div>
            """, unsafe_allow_html=True)

            time.sleep(1)
            progress_bar.progress(100)

            # Display the results with enhanced styling
            st.markdown("""
            <h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 3rem;">
                <span style="color: #5BC0BE;">📊</span> Verification Results
            </h2>
            """, unsafe_allow_html=True)

            # Verdict card with dark background and white text
            verdict_class = result["verdict"].lower() if result["verdict"] in ["Real", "Fake"] else "unknown"

            # Get the real and fake percentages
            real_percentage = result.get("real_percentage", 50)
            fake_percentage = result.get("fake_percentage", 50)

            # Determine color based on percentage (gradient from red to green)
            # The more real, the greener; the more fake, the redder
            if real_percentage >= 75:  # Strongly real
                verdict_color = "#4ade80"  # Green
                verdict_icon = "✅"
            elif real_percentage >= 60:  # Moderately real
                verdict_color = "#a3e635"  # Light green
                verdict_icon = "✅"
            elif real_percentage >= 40:  # Mixed/Uncertain
                verdict_color = "#fde047"  # Yellow
                verdict_icon = "⚠️"
            elif real_percentage >= 25:  # Moderately fake
                verdict_color = "#fb923c"  # Orange
                verdict_icon = "❌"
            else:  # Strongly fake
                verdict_color = "#f87171"  # Red
                verdict_icon = "❌"

            # For backward compatibility
            if verdict_icon == "❓":
                verdict_icon = "⚠️"

            st.markdown(f"""
            <div class="verification-card" style="background-color: #1e293b; color: white; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); padding: 1.5rem; border-radius: 8px;">
                <div style="display: flex; align-items: center; gap: 1rem; background-color: #0f172a;
                            padding: 1.25rem; border-radius: 8px; margin-bottom: 1.25rem; border-left: 5px solid {verdict_color};">
                    <div style="font-size: 3rem;">{verdict_icon}</div>
                    <div style="flex-grow: 1;">
                        <h2 style="margin-top: 0; color: {verdict_color}; font-size: 2rem !important; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
                            Content Analysis
                        </h2>
                        <div style="display: flex; align-items: center; gap: 1rem; margin-top: 0.5rem;">
                            <div style="flex-grow: 1;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                    <span style="color: #f87171; font-weight: 600;">Fake: {fake_percentage}%</span>
                                    <span style="color: #4ade80; font-weight: 600;">Real: {real_percentage}%</span>
                                </div>
                                <div style="height: 10px; background-color: #334155; border-radius: 5px; overflow: hidden; margin-top: 0.25rem;">
                                    <div style="width: {real_percentage}%; height: 100%; background: linear-gradient(to right, #f87171, #fde047, #4ade80); border-radius: 5px;"></div>
                                </div>
                            </div>
                        </div>
                        <p style="color: #ffffff; font-weight: 600; margin-top: 0.75rem; margin-bottom: 0; font-size: 1.1rem;">
                            Confidence: {result["confidence"].upper()}
                        </p>
                    </div>
                </div>
                <div class="confidence-meter" style="height: 12px; background-color: #334155; margin: 1.5rem 0; border-radius: 6px;">
                    <div class="fill" style="width: {100 if result["confidence"].upper() == "HIGH" else 70 if result["confidence"].upper() == "MEDIUM" else 40}%;
                         background: {verdict_color}; height: 12px; border-radius: 6px;"></div>
                </div>
                <div style="background-color: #ffffff; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;
                            border-left: 5px solid {verdict_color}; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                    <h3 style="margin-top: 0; color: #000000; margin-bottom: 0.75rem;">EXPLANATION:</h3>
                    <p style="color: #000000; margin: 0; font-weight: 500; line-height: 1.6; font-size: 1.05rem;">
                        {result["analysis"]}
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Display model comparison results if available
            if 'comparison_results' in st.session_state:
                comp = st.session_state.comparison_results
                primary_model = comp["primary_model"]
                comparison_model = comp["comparison_model"]
                primary_real = comp["primary_real"]
                comparison_real = comp["comparison_real"]
                real_difference = comp["real_difference"]

                # Format model names for display
                primary_display = primary_model.split(":")[0] if ":" in primary_model else primary_model
                comparison_display = comparison_model.split(":")[0] if ":" in comparison_model else comparison_model

                # Determine agreement level
                if real_difference < 10:
                    agreement_level = "High Agreement"
                    agreement_color = "#4ade80"  # Green
                elif real_difference < 20:
                    agreement_level = "Moderate Agreement"
                    agreement_color = "#fde047"  # Yellow
                else:
                    agreement_level = "Low Agreement"
                    agreement_color = "#f87171"  # Red

                st.markdown("""
                <h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 3rem;">
                    <span style="color: #5BC0BE;">🤖</span> Model Comparison
                </h2>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background-color: #1e293b; color: white; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
                    <div style="display: flex; align-items: center; gap: 1rem; background-color: #0f172a;
                                padding: 1.25rem; border-radius: 8px; margin-bottom: 1.25rem; border-left: 5px solid {agreement_color};">
                        <div style="flex-grow: 1;">
                            <h3 style="margin-top: 0; color: white; font-size: 1.5rem !important; margin-bottom: 1rem;">
                                Model Agreement: <span style="color: {agreement_color};">{agreement_level}</span>
                            </h3>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <div style="font-weight: 600; font-size: 1.1rem;">{primary_display}</div>
                                <div style="font-weight: 600; font-size: 1.1rem;">{comparison_display}</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <div style="color: #4ade80; font-weight: 600;">Real: {primary_real}%</div>
                                <div style="color: #4ade80; font-weight: 600;">Real: {comparison_real}%</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <div style="color: #f87171; font-weight: 600;">Fake: {100-primary_real}%</div>
                                <div style="color: #f87171; font-weight: 600;">Fake: {100-comparison_real}%</div>
                            </div>
                            <div style="height: 10px; background-color: #334155; border-radius: 5px; overflow: hidden; margin-top: 1rem;">
                                <div style="width: {min(100, real_difference*2)}%; height: 100%; background-color: {agreement_color}; border-radius: 5px;"></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                                <div style="font-size: 0.9rem; opacity: 0.8;">Agreement</div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Difference: {real_difference}%</div>
                            </div>
                        </div>
                    </div>
                    <p style="color: #000000; margin: 0; font-weight: 500; line-height: 1.6; font-size: 1.05rem; background-color: #ffffff; padding: 1rem; border-radius: 6px;">
                        The models {primary_display} and {comparison_display} {
                        "largely agree" if real_difference < 10 else
                        "somewhat disagree" if real_difference < 20 else
                        "significantly disagree"} on the credibility of this content.
                        {primary_display} rates it as {primary_real}% real, while {comparison_display} rates it as {comparison_real}% real.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Key claims with dark background and white text
            st.markdown("""
            <div style="background-color: #1e293b; color: white; padding: 1.5rem; border-radius: 8px; margin-top: 2rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
                <h3 style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0; color: #ffffff;
                           border-bottom: 2px solid #5BC0BE; padding-bottom: 0.75rem; margin-bottom: 1.25rem;">
                    <span style="color: #5BC0BE; font-size: 1.5rem;">🔍</span> Key Claims Analyzed
                </h3>
                <p style="color: #000000; margin-bottom: 1.5rem; font-weight: 500; background-color: #ffffff;
                          padding: 1rem; border-radius: 6px; border-left: 3px solid #5BC0BE;">
                    The LLM identified and verified the following key claims from the article:
                </p>
            """, unsafe_allow_html=True)

            for i, claim in enumerate(result["claims"], 1):
                st.markdown(f"""
                <div style="background-color: #ffffff; padding: 1.25rem; border-radius: 8px; margin-bottom: 1rem;
                            border-left: 4px solid #5BC0BE; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                    <div style="display: flex; gap: 1rem; align-items: flex-start;">
                        <div style="background-color: #5BC0BE; color: white; min-width: 32px; height: 32px; border-radius: 50%;
                                    display: flex; align-items: center; justify-content: center; font-size: 1rem; font-weight: bold;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);">{i}</div>
                        <div style="color: #000000; font-weight: 500; font-size: 1.05rem; line-height: 1.6;">{claim}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            </div>
            """, unsafe_allow_html=True)

            # Search results with dark background and white text
            st.markdown("""
            <div style="background-color: #1e293b; color: white; padding: 1.5rem; border-radius: 8px; margin-top: 2rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
                <h3 style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0; color: #ffffff;
                           border-bottom: 2px solid #5BC0BE; padding-bottom: 0.75rem; margin-bottom: 1.25rem;">
                    <span style="color: #5BC0BE; font-size: 1.5rem;">🌐</span> Web Search Results
                </h3>
                <p style="color: #000000; margin-bottom: 1.5rem; font-weight: 500; background-color: #ffffff;
                          padding: 1rem; border-radius: 6px; border-left: 3px solid #5BC0BE;">
                    The following search results were used to verify the claims:
                </p>
            """, unsafe_allow_html=True)

            # Search results in expandable cards with better visibility
            for i, search_result in enumerate(result["search_results"], 1):
                with st.expander(f"Search Results for Claim {i}", expanded=False):
                    # Use a container for better styling control
                    search_container = st.container()

                    # Display the claim header
                    search_container.markdown("""
                    <div style="background-color: #1e293b; color: white; padding: 1.25rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); margin-bottom: 1rem;">
                        <h4 style="margin-top: 0; color: #ffffff; font-weight: 600; display: flex; align-items: center; gap: 0.75rem;">
                            <span style="background-color: #5BC0BE; color: white; width: 24px; height: 24px; border-radius: 50%;
                                        display: flex; align-items: center; justify-content: center; font-size: 0.9rem; font-weight: bold;">
                                C
                            </span>
                            Claim:
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display the claim in a separate markdown component
                    search_container.markdown(f"""
                    <div style="background-color: #ffffff; padding: 1rem; border-radius: 8px; margin-bottom: 1.25rem;
                                border-left: 4px solid #5BC0BE; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                        <p style="color: #000000; margin: 0; font-weight: 500; font-size: 1.05rem; line-height: 1.6;">
                            {html.escape(search_result["claim"])}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display the search results header
                    search_container.markdown("""
                    <div style="background-color: #1e293b; color: white; padding: 1.25rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); margin-bottom: 1rem;">
                        <h4 style="margin-top: 0; color: #ffffff; font-weight: 600; display: flex; align-items: center; gap: 0.75rem;">
                            <span style="background-color: #5BC0BE; color: white; width: 24px; height: 24px; border-radius: 50%;
                                        display: flex; align-items: center; justify-content: center; font-size: 0.9rem; font-weight: bold;">
                                S
                            </span>
                            Search Results:
                        </h4>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display search results in a clean format
                    search_results_text = search_result["search_results"]

                    # Format the search results as a list
                    if isinstance(search_results_text, str):
                        # Split the search results by numbered points if they exist
                        results_list = []
                        lines = search_results_text.split('\n')
                        current_result = ""

                        for line in lines:
                            if re.match(r'^\d+\.', line.strip()):
                                if current_result:
                                    results_list.append(current_result.strip())
                                current_result = line
                            else:
                                current_result += " " + line

                        if current_result:
                            results_list.append(current_result.strip())

                        # If we couldn't split by numbers, just use the whole text
                        if not results_list:
                            results_list = [search_results_text]

                        # Display each result in a styled container
                        for result_item in results_list:
                            # Use Streamlit's native text display to avoid HTML rendering issues
                            # First, create a styled container
                            search_container.markdown("""
                            <div style="background-color: #ffffff; padding: 1.25rem; border-radius: 8px;
                                        border-left: 4px solid #5BC0BE; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                                        margin-bottom: 0.75rem; color: #000000; font-size: 1rem; line-height: 1.6;">
                            """, unsafe_allow_html=True)

                            # Then display the text content directly using Streamlit's text component
                            # This avoids HTML rendering issues completely
                            search_container.text(result_item)

                            # Close the styled container
                            search_container.markdown("</div>", unsafe_allow_html=True)
                    else:
                        # Display a message if no search results are available
                        # Use the same approach as for results to maintain consistency
                        search_container.markdown("""
                        <div style="background-color: #ffffff; padding: 1.25rem; border-radius: 8px;
                                    border-left: 4px solid #5BC0BE; box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                                    margin-bottom: 0.75rem; color: #000000; font-size: 1rem; line-height: 1.6;">
                        """, unsafe_allow_html=True)

                        search_container.text("No search results available")

                        search_container.markdown("</div>", unsafe_allow_html=True)

            # Add a conclusion section with dark background and white text
            st.markdown("""
            </div>
            <div style="background-color: #1e293b; color: white; padding: 1.75rem; border-radius: 8px; margin-top: 2rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
                <h3 style="margin-top: 0; color: #ffffff; display: flex; align-items: center; gap: 0.75rem;
                           border-bottom: 2px solid #5BC0BE; padding-bottom: 0.75rem; margin-bottom: 1.25rem;">
                    <span style="color: #5BC0BE; font-size: 1.5rem;">📝</span> Conclusion
                </h3>
                <div style="background-color: #ffffff; padding: 1.25rem; border-radius: 8px;
                            border-left: 4px solid #5BC0BE; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                    <p style="color: #000000; margin-bottom: 0; font-weight: 500; font-size: 1.05rem; line-height: 1.6;">
                        This verification was performed using a local LLM and web search. While the results provide a good indication
                        of the article's credibility, always verify important information with multiple trusted sources.
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            error_msg = str(e)

            # Check for common error patterns
            if "Extra data" in error_msg and "char" in error_msg:
                error_title = "JSON Parsing Error"
                error_details = """
                There was an error processing the search results. This is likely due to an issue with the web search functionality.
                We've implemented fixes that should resolve this issue. Please try again with your article.
                """
                troubleshooting = [
                    "Try verifying the article again",
                    "Try a different article with clear factual claims",
                    "Check your internet connection",
                    f"Verify that the selected model ({model_name}) is available"
                ]
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                error_title = "Connection Error"
                error_details = """
                There was an error connecting to the search service or the connection timed out.
                This could be due to network issues or the search service being temporarily unavailable.
                """
                troubleshooting = [
                    "Check your internet connection",
                    "Try again in a few minutes",
                    "Try a different article"
                ]
            elif "ollama" in error_msg.lower():
                error_title = "Ollama Error"
                error_details = """
                There was an error communicating with the Ollama service.
                """
                troubleshooting = [
                    "Check that Ollama is running",
                    f"Verify that the selected model ({model_name}) is available",
                    f"Try pulling the model again with: <code>ollama pull {model_name}</code>",
                    "Restart Ollama and refresh this page"
                ]
            else:
                error_title = "Verification Error"
                error_details = "An unexpected error occurred during the verification process."
                troubleshooting = [
                    "Try verifying the article again",
                    "Check that Ollama is running",
                    f"Verify that the selected model ({model_name}) is available",
                    "Try a different article"
                ]

            # Enhanced error message with dark background and white text
            st.markdown(f"""
            <div style="background-color: #1e293b; color: white; padding: 1.75rem; border-radius: 8px; margin: 2rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
                <h3 style="margin-top: 0; color: #f87171; display: flex; align-items: center; gap: 0.75rem;
                           border-bottom: 2px solid #f87171; padding-bottom: 0.75rem; margin-bottom: 1.25rem;">
                    <span style="font-size: 1.5rem;">❌</span> {error_title}
                </h3>
                <div style="background-color: #0f172a; padding: 1.25rem; border-radius: 8px;
                            border-left: 4px solid #f87171; box-shadow: 0 2px 5px rgba(0,0,0,0.2); margin-bottom: 1.5rem;">
                    <p style="color: #e2e8f0; margin-bottom: 0; font-weight: 500; font-size: 1.05rem; line-height: 1.6;">
                        {error_details}
                    </p>
                </div>
                <div style="background-color: #0f172a; padding: 1.25rem; border-radius: 8px; font-family: monospace;
                            font-size: 0.95rem; border-left: 4px solid #f87171; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                    <pre style="color: #f87171; margin: 0; white-space: pre-wrap; line-height: 1.6;">{error_msg}</pre>
                </div>
            </div>

            <div style="background-color: #1e293b; color: white; padding: 1.75rem; border-radius: 8px; margin: 2rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
                <h3 style="margin-top: 0; color: #e2e8f0; display: flex; align-items: center; gap: 0.75rem;
                           border-bottom: 2px solid #5BC0BE; padding-bottom: 0.75rem; margin-bottom: 1.25rem;">
                    <span style="color: #5BC0BE; font-size: 1.5rem;">🔧</span> Troubleshooting Steps
                </h3>
                <div style="background-color: #0f172a; padding: 1.25rem; border-radius: 8px;
                            border-left: 4px solid #5BC0BE; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
                    <ul style="color: #e2e8f0; margin-bottom: 0; padding-left: 1.5rem; font-weight: 500; font-size: 1.05rem; line-height: 1.6;">
                        {' '.join(f'<li style="margin-bottom: 0.5rem;">{step}</li>' for step in troubleshooting)}
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Information about the feature with dark background and white text
with st.expander("About LLM Verification", expanded=False):
    # Use separate markdown components to avoid rendering issues
    st.markdown("""
    <div style="padding: 1.75rem; background-color: #1e293b; color: white; border-radius: 8px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
        <h3 style="color: #ffffff; margin-top: 0; display: flex; align-items: center; gap: 0.75rem;
                   border-bottom: 2px solid #5BC0BE; padding-bottom: 0.75rem; margin-bottom: 1.25rem;">
            <span style="color: #5BC0BE; font-size: 1.5rem;">🤖</span> How LLM Verification Works
        </h3>
    """, unsafe_allow_html=True)

    st.markdown("""
        <p style="color: #ffffff; margin-bottom: 1.5rem; font-weight: 500; background-color: #000000;
                  padding: 1rem; border-radius: 6px; border-left: 3px solid #5BC0BE; font-size: 1.05rem; line-height: 1.6;">
            This feature uses a locally running Large Language Model (LLM) combined with web search to verify news articles:
        </p>
    """, unsafe_allow_html=True)

    # Step 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background-color: #000000; border-radius: 8px; padding: 1.25rem;
                    border-left: 4px solid #5BC0BE; box-shadow: 0 2px 5px rgba(0,0,0,0.2); height: 100%;">
            <div style="display: flex; gap: 0.75rem; align-items: center; margin-bottom: 0.75rem;">
                <div style="background-color: #5BC0BE; color: white; width: 32px; height: 32px; border-radius: 50%;
                            display: flex; align-items: center; justify-content: center; font-size: 1rem; font-weight: bold;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.2);">1</div>
                <h4 style="margin: 0; color: #ffffff; font-size: 1.1rem;">Key Claim Extraction</h4>
            </div>
            <p style="color: #ffffff; margin: 0; padding-left: 3rem; font-size: 1rem; line-height: 1.6;">
                The LLM identifies the main factual claims in the article
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Step 2
    with col2:
        st.markdown("""
        <div style="background-color: #000000; border-radius: 8px; padding: 1.25rem;
                    border-left: 4px solid #5BC0BE; box-shadow: 0 2px 5px rgba(0,0,0,0.2); height: 100%;">
            <div style="display: flex; gap: 0.75rem; align-items: center; margin-bottom: 0.75rem;">
                <div style="background-color: #5BC0BE; color: white; width: 32px; height: 32px; border-radius: 50%;
                            display: flex; align-items: center; justify-content: center; font-size: 1rem; font-weight: bold;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.2);">2</div>
                <h4 style="margin: 0; color: #ffffff; font-size: 1.1rem;">Web Search</h4>
            </div>
            <p style="color: #ffffff; margin: 0; padding-left: 3rem; font-size: 1rem; line-height: 1.6;">
                Each claim is searched on the web to find supporting or contradicting information
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Step 3 and 4
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div style="background-color: #000000; border-radius: 8px; padding: 1.25rem;
                    border-left: 4px solid #5BC0BE; box-shadow: 0 2px 5px rgba(0,0,0,0.2); height: 100%;">
            <div style="display: flex; gap: 0.75rem; align-items: center; margin-bottom: 0.75rem;">
                <div style="background-color: #5BC0BE; color: white; width: 32px; height: 32px; border-radius: 50%;
                            display: flex; align-items: center; justify-content: center; font-size: 1rem; font-weight: bold;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.2);">3</div>
                <h4 style="margin: 0; color: #ffffff; font-size: 1.1rem;">Analysis</h4>
            </div>
            <p style="color: #ffffff; margin: 0; padding-left: 3rem; font-size: 1rem; line-height: 1.6;">
                The LLM analyzes the search results and compares them with the article
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background-color: #000000; border-radius: 8px; padding: 1.25rem;
                    border-left: 4px solid #5BC0BE; box-shadow: 0 2px 5px rgba(0,0,0,0.2); height: 100%;">
            <div style="display: flex; gap: 0.75rem; align-items: center; margin-bottom: 0.75rem;">
                <div style="background-color: #5BC0BE; color: white; width: 32px; height: 32px; border-radius: 50%;
                            display: flex; align-items: center; justify-content: center; font-size: 1rem; font-weight: bold;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.2);">4</div>
                <h4 style="margin: 0; color: #ffffff; font-size: 1.1rem;">Verdict</h4>
            </div>
            <p style="color: #ffffff; margin: 0; padding-left: 3rem; font-size: 1rem; line-height: 1.6;">
                Based on the analysis, the LLM provides a verdict on whether the article appears to be real or fake
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

    # Privacy and Limitations sections
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("""
        <div style="background-color: #0f172a; border-radius: 8px; padding: 1.25rem;
                    border-left: 4px solid #5BC0BE; box-shadow: 0 2px 5px rgba(0,0,0,0.2); height: 100%;">
            <h3 style="color: #ffffff; margin-top: 0; display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <span style="color: #5BC0BE; font-size: 1.25rem;">🔒</span> Privacy & Performance
            </h3>
        """, unsafe_allow_html=True)

        st.markdown("""
            <ul style="color: #000000; padding-left: 1.5rem; font-size: 1rem; line-height: 1.6; background-color: #ffffff; border-radius: 6px; padding: 1rem;">
                <li style="margin-bottom: 0.5rem;">All LLM processing happens locally on your machine</li>
                <li style="margin-bottom: 0.5rem;">No article data is sent to external AI services</li>
                <li style="margin-bottom: 0.5rem;">Performance depends on your hardware and the model size</li>
                <li>First-time model loading may take a few moments</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div style="background-color: #0f172a; border-radius: 8px; padding: 1.25rem;
                    border-left: 4px solid #5BC0BE; box-shadow: 0 2px 5px rgba(0,0,0,0.2); height: 100%;">
            <h3 style="color: #e2e8f0; margin-top: 0; display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
                <span style="color: #5BC0BE; font-size: 1.25rem;">⚠️</span> Limitations
            </h3>
        """, unsafe_allow_html=True)

        st.markdown("""
            <ul style="color: #000000; padding-left: 1.5rem; font-size: 1rem; line-height: 1.6; background-color: #ffffff; border-radius: 6px; padding: 1rem;">
                <li style="margin-bottom: 0.5rem;">The system is not perfect and should be used as one tool among many for fact-checking</li>
                <li style="margin-bottom: 0.5rem;">Results depend on the quality of web search results</li>
                <li style="margin-bottom: 0.5rem;">The LLM may occasionally hallucinate or make errors in its analysis</li>
                <li>Always verify important information with multiple trusted sources</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Close the main container div
    st.markdown("</div>", unsafe_allow_html=True)
