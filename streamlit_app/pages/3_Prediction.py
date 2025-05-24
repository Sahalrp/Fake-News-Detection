import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lime.lime_text import LimeTextExplainer
import joblib
import os
from pathlib import Path
from textblob import TextBlob
import re
import sys
import time

# Add parent directory to path to import html_renderer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from html_renderer import render_html, render_styled_html, render_status_indicator, render_button_group

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import clean_text, extract_stylometric_features, extract_sentiment_features, load_models, check_ollama_status, load_llm_verifier

# Add the parent directory to the path so we can import the llm_verification module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Try to import the LLM verification module
try:
    from llm_verification import NewsVerifier, OllamaClient
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Page config
st.set_page_config(page_title="News Prediction", page_icon="🎯", layout="wide")

# Add parent directory to path to import shared styles
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_styles import load_css

# Load shared CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Additional custom CSS to match the main app styling
st.markdown("""
<style>
    /* Import the same font as the main app */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Ensure all text is visible with proper contrast */
    p, span, div, li, a, label, text {
        color: #333333 !important;
    }

    /* Prediction cards with modern styling */
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 5px solid #5BC0BE;
        color: #3A506B !important;
    }

    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }

    .prediction-card.real {
        border-top: 5px solid #2E7D32;
        background: linear-gradient(to bottom, rgba(46, 125, 50, 0.05), white);
    }

    .prediction-card.fake {
        border-top: 5px solid #C62828;
        background: linear-gradient(to bottom, rgba(198, 40, 40, 0.05), white);
    }

    .prediction-card h2 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }

    .prediction-card h3 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #6c757d !important;
        margin-bottom: 1rem !important;
    }

    .prediction-card p {
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Confidence meter with animation */
    .confidence-meter {
        height: 10px;
        background: #f0f0f0;
        border-radius: 5px;
        overflow: hidden;
        margin-top: 1rem;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .confidence-meter .fill {
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

    /* Alert styling to match main app */
    .st-success {
        background-color: rgba(91, 192, 190, 0.2) !important;
        border: 1px solid #5BC0BE !important;
        color: #2c7c7a !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    .st-info {
        background-color: rgba(58, 80, 107, 0.2) !important;
        border: 1px solid #3A506B !important;
        color: #2d3748 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    .st-warning {
        background-color: rgba(255, 209, 102, 0.2) !important;
        border: 1px solid #FFD166 !important;
        color: #b7791f !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    .st-error {
        background-color: rgba(255, 107, 107, 0.2) !important;
        border: 1px solid #FF6B6B !important;
        color: #c53030 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    /* LLM verification styling */
    .verification-card {
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }

    .verification-card:hover {
        transform: translateY(-5px);
    }

    .verification-card h2 {
        margin-bottom: 1rem !important;
        font-weight: 700 !important;
    }

    .real {
        background: linear-gradient(to bottom right, rgba(46, 125, 50, 0.05), white);
        border-left: 5px solid #2E7D32;
    }

    .fake {
        background: linear-gradient(to bottom right, rgba(198, 40, 40, 0.05), white);
        border-left: 5px solid #C62828;
    }

    .unknown {
        background: linear-gradient(to bottom right, rgba(255, 209, 102, 0.05), white);
        border-left: 5px solid #FFD166;
    }

    /* Claim box styling */
    .claim-box {
        background-color: white;
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

    /* Text input styling */
    textarea {
        border-radius: 8px !important;
        border: 1px solid #e0e0e0 !important;
        padding: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }

    textarea:focus {
        border-color: #5BC0BE !important;
        box-shadow: 0 0 0 2px rgba(91, 192, 190, 0.2) !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: #5BC0BE !important;
        color: white !important;
        font-weight: 500 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
    }

    .stButton > button:hover {
        background-color: #3A506B !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
    }

    /* Primary button */
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background-color: #FF6B6B !important;
    }

    .stButton > button[data-baseweb="button"][kind="primary"]:hover {
        background-color: #e05e5e !important;
    }

    /* Section headers */
    h1, h2, h3 {
        color: #3A506B !important;
    }

    h1 {
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #5BC0BE !important;
    }

    h2 {
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        font-weight: 500 !important;
        font-size: 1.4rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #3A506B !important;
        background-color: rgba(91, 192, 190, 0.1) !important;
        border-radius: 8px !important;
    }

    .streamlit-expanderContent {
        border-left: 1px solid #5BC0BE !important;
        padding-left: 1rem !important;
        margin-left: 0.5rem !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background-color: #5BC0BE !important;
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

# Get root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get prediction color based on confidence
def get_prediction_color(confidence):
    if confidence >= 0.8:
        return "#2E7D32"  # Strong confidence - dark green
    elif confidence >= 0.6:
        return "#1E88E5"  # Medium confidence - blue
    else:
        return "#FB8C00"  # Low confidence - orange

# Sidebar with improved styling
with st.sidebar:
    # Logo and title
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://img.icons8.com/fluency/96/000000/target.png", width=60)
    with col2:
        st.markdown("<h2 style='margin-top:0.5rem;'>News<br>Prediction</h2>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Check if Ollama is available for LLM verification
    ollama_status = check_ollama_status()

    # Model settings section
    st.markdown("### 🧠 Model Settings")

    # Add explanation about improved fake news detection
    st.markdown("""
    <div style="background-color: rgba(91, 192, 190, 0.1); padding: 0.75rem; border-radius: 4px; margin-bottom: 1rem;">
        <p style="margin: 0; font-size: 0.9rem; color: white;">
            <span style="font-weight: 600;">💡 Enhanced Detection:</span>
            Our system now uses multiple methods to identify fake news:
        </p>
        <ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem; font-size: 0.85rem; color: rgba(255,255,255,0.9);">
            <li>Machine learning models analyze writing patterns</li>
            <li>Factual accuracy checking with LLM (when Ollama is available)</li>
            <li>Adjustable sensitivity for different content types</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Model selection with improved styling
    model_type = st.selectbox(
        "Primary Model",
        ["Logistic Regression", "Random Forest"],
        index=0,
        help="Choose which model to prioritize for prediction"
    )

    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Fake News Sensitivity",
        min_value=0.3,
        max_value=0.7,
        value=0.4,
        step=0.05,
        help="Lower values make the system more sensitive to potential fake news (recommended: 0.4)"
    )

    # LLM verification section
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🤖 LLM Verification")

    # Ollama status with custom indicator
    if ollama_status:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div class="status-indicator success"></div>
            <div>Ollama is running</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: rgba(255,255,255,0.1); padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem; font-size: 0.9rem;">
            You can use LLM verification for additional fact-checking of the article.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div class="status-indicator error"></div>
            <div>Ollama not running</div>
        </div>

        <div style="background-color: rgba(198, 40, 40, 0.2); padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem; font-size: 0.9rem;">
            Install Ollama to enable LLM verification with internet search.
            <div style="margin-top: 0.5rem;">
                <a href="/LLM_Verification" style="color: white; text-decoration: underline;">Learn more</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Add navigation section
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📊 Navigation")

    # Navigation buttons
    nav_buttons = [
        {
            'label': 'Home',
            'url': '/',
            'icon': '🏠'
        },
        {
            'label': 'LLM Verify',
            'url': '/LLM_Verification',
            'icon': '🔍'
        }
    ]

    render_button_group(nav_buttons, height=60)

# Page title with animation
title_html = """
<div class="page-title">
    <h1 style="font-size: 2.8rem; font-weight: 700; margin-bottom: 0.5rem; color: white;">🎯 News Prediction</h1>
    <p style="font-size: 1.2rem; color: rgba(255,255,255,0.8); max-width: 800px; margin: 0 auto;">
        Enter a news article to analyze its authenticity using our advanced machine learning models
    </p>
</div>
"""
render_styled_html(title_html, height=120)

# Introduction card
intro_html = """
<div style="background: linear-gradient(135deg, rgba(58, 80, 107, 0.05), rgba(91, 192, 190, 0.05));
            padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; animation: slideIn 0.8s ease-out;">
    <div style="display: flex; align-items: flex-start; gap: 1rem;">
        <div style="font-size: 2.5rem; color: #5BC0BE; margin-top: 0.5rem;">💡</div>
        <div>
            <h3 style="margin-top: 0; margin-bottom: 0.5rem; color: white;">How It Works</h3>
            <p style="color: rgba(255,255,255,0.8); margin-bottom: 0.5rem;">
                Our system uses multiple machine learning models to analyze news articles and determine if they're likely to be real or fake.
                We examine writing style, sentiment, and content patterns to make predictions.
            </p>
            <p style="color: rgba(255,255,255,0.8); margin-bottom: 0;">
                For best results, paste a complete news article with at least several paragraphs.
            </p>
        </div>
    </div>
</div>

<style>
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}
</style>
"""
render_html(intro_html, height=180)

# Load models with spinner
with st.spinner("Loading models..."):
    lr_model, rf_model, vectorizer = load_models(ROOT_DIR)

if lr_model and rf_model and vectorizer:
    # === DIAGNOSTIC BLOCK: Test Model on Known Real and Fake (hidden in expander) ===
    with st.expander("🛠️ Model Diagnostic (for developers)", expanded=False):
        try:
            fake_sample = pd.read_csv(os.path.join(ROOT_DIR, "2_Data_Collection/data/Fake.csv"), nrows=1)
            true_sample = pd.read_csv(os.path.join(ROOT_DIR, "2_Data_Collection/data/True.csv"), nrows=1)
            fake_text = fake_sample.iloc[0]['text']
            true_text = true_sample.iloc[0]['text']

            # Vectorize
            fake_vec = vectorizer.transform([fake_text])
            true_vec = vectorizer.transform([true_text])

            # Predict with adjusted logic for better fake news detection
            fake_proba_raw = lr_model.predict_proba(fake_vec)[0]
            true_proba_raw = lr_model.predict_proba(true_vec)[0]

            # Apply the same adjusted prediction logic with the user-selected threshold
            fake_pred = 0 if fake_proba_raw[0] > confidence_threshold else 1
            true_pred = 0 if true_proba_raw[0] > confidence_threshold else 1

            # Use raw probabilities for display
            fake_proba = fake_proba_raw
            true_proba = true_proba_raw

            # Display in a more structured way
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div style="background-color: rgba(255, 107, 107, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #FF6B6B;">
                    <h4 style="margin-top: 0; color: #C62828;">Fake Sample Test</h4>
                """, unsafe_allow_html=True)
                st.write(f"Prediction: {'Fake' if fake_pred == 0 else 'Real'}")
                st.write(f"Probability: {fake_proba[0]:.2f} (Fake), {fake_proba[1]:.2f} (Real)")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div style="background-color: rgba(46, 125, 50, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #2E7D32;">
                    <h4 style="margin-top: 0; color: #2E7D32;">Real Sample Test</h4>
                """, unsafe_allow_html=True)
                st.write(f"Prediction: {'Fake' if true_pred == 0 else 'Real'}")
                st.write(f"Probability: {true_proba[0]:.2f} (Fake), {true_proba[1]:.2f} (Real)")
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"Diagnostic failed: {e}")

    # Text input with improved styling
    st.markdown("""
    <div style="margin: 2rem 0 1rem 0;">
        <h3 style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
            <span style="color: #5BC0BE;">📝</span> Enter News Article
        </h3>
    </div>
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
        "Paste or type news text here:",
        height=200,
        help="Paste the complete news article text you want to analyze",
        placeholder="Paste your news article here... (For best results, include the full article with headline and several paragraphs)"
    )

    # Add analyze button with icon
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Article",
            help="Click to analyze the news article",
            use_container_width=True
        )

    # Preprocess input text for consistency with training
    cleaned_text_input = clean_text(text_input)

    if cleaned_text_input and (analyze_button or 'analyze_clicked' in st.session_state):
        # Store that analysis was clicked
        if analyze_button:
            st.session_state.analyze_clicked = True

        # Show progress bar for analysis steps
        progress_bar = st.progress(0)

        with st.spinner("Analyzing article..."):
            try:
                # Step 1: Extract features
                progress_bar.progress(10)
                st.markdown("""
                <div style="margin-top: 1rem; font-size: 0.9rem; color: #6c757d;">
                    <span style="color: #5BC0BE;">⏳</span> Extracting stylometric features...
                </div>
                """, unsafe_allow_html=True)

                style_features = extract_stylometric_features(cleaned_text_input)
                progress_bar.progress(20)

                st.markdown("""
                <div style="font-size: 0.9rem; color: #6c757d;">
                    <span style="color: #5BC0BE;">⏳</span> Analyzing sentiment patterns...
                </div>
                """, unsafe_allow_html=True)

                sentiment_features = extract_sentiment_features(cleaned_text_input)
                progress_bar.progress(30)

                # Step 2: Combine features
                features = {**style_features, **sentiment_features}
                features_df = pd.DataFrame([features])

                st.markdown("""
                <div style="font-size: 0.9rem; color: #6c757d;">
                    <span style="color: #5BC0BE;">⏳</span> Vectorizing text content...
                </div>
                """, unsafe_allow_html=True)

                # Step 3: Get text features
                text_features = vectorizer.transform([cleaned_text_input])
                progress_bar.progress(50)

                st.markdown("""
                <div style="font-size: 0.9rem; color: #6c757d;">
                    <span style="color: #5BC0BE;">⏳</span> Running prediction models...
                </div>
                """, unsafe_allow_html=True)

                # Step 4: Get predictions and probabilities
                # The models were trained with 0=Fake, 1=Real
                # But we need to make sure we're correctly identifying fake news

                # Get raw predictions
                lr_pred_raw = lr_model.predict(text_features)[0]
                rf_pred_raw = rf_model.predict(text_features)[0]

                # Get probabilities
                lr_proba_raw = lr_model.predict_proba(text_features)[0]
                rf_proba_raw = rf_model.predict_proba(text_features)[0]

                # Adjust prediction logic to better detect fake news
                # Use the user-selected threshold for fake news sensitivity
                # Lower threshold = more sensitive to fake news
                lr_pred = 0 if lr_proba_raw[0] > confidence_threshold else 1
                rf_pred = 0 if rf_proba_raw[0] > confidence_threshold else 1

                # Use the raw probabilities for display
                lr_proba = lr_proba_raw
                rf_proba = rf_proba_raw

                progress_bar.progress(70)

                # Step 5: Generate explanations
                st.markdown("""
                <div style="font-size: 0.9rem; color: #6c757d;">
                    <span style="color: #5BC0BE;">⏳</span> Generating explanations...
                </div>
                """, unsafe_allow_html=True)

                # Complete the progress
                progress_bar.progress(100)

                # Add a small delay for visual effect
                time.sleep(0.5)

                # Remove progress elements
                progress_bar.empty()

                # Display results with animation
                st.markdown("""
                <div style="text-align: center; margin: 2rem 0; animation: fadeIn 0.8s ease-in-out;">
                    <h2 style="font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;">
                        📊 Analysis Results
                    </h2>
                    <p style="color: #6c757d; max-width: 600px; margin: 0 auto 2rem auto;">
                        Our models have analyzed the article and generated the following predictions
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Display predictions in a more visually appealing way
                col1, col2 = st.columns(2)

                # Logistic Regression with improved card
                with col1:
                    lr_confidence = lr_proba.max()
                    lr_class = "Real" if lr_pred == 1 else "Fake"
                    lr_icon = "✅" if lr_pred == 1 else "❌"
                    lr_color = "#2E7D32" if lr_pred == 1 else "#C62828"

                    st.markdown(f"""
                    <div class="prediction-card {'real' if lr_pred == 1 else 'fake'}" style="animation: slideIn 0.5s ease-out;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h3 style="margin: 0;">Logistic Regression</h3>
                            <div style="font-size: 0.9rem; color: #6c757d; background-color: rgba(0,0,0,0.05); padding: 0.25rem 0.5rem; border-radius: 4px;">
                                Primary Model
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                            <div style="font-size: 2.5rem;">{lr_icon}</div>
                            <div>
                                <h2 style="margin: 0; color: {lr_color};">{lr_class}</h2>
                                <p style="margin: 0; color: #6c757d;">with {lr_confidence:.1%} confidence</p>
                            </div>
                        </div>
                        <div class="confidence-meter">
                            <div class="fill" style="width: {lr_confidence*100}%; background: {get_prediction_color(lr_confidence)}"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Random Forest with improved card
                with col2:
                    rf_confidence = rf_proba.max()
                    rf_class = "Real" if rf_pred == 1 else "Fake"
                    rf_icon = "✅" if rf_pred == 1 else "❌"
                    rf_color = "#2E7D32" if rf_pred == 1 else "#C62828"

                    st.markdown(f"""
                    <div class="prediction-card {'real' if rf_pred == 1 else 'fake'}" style="animation: slideIn 0.7s ease-out;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                            <h3 style="margin: 0;">Random Forest</h3>
                            <div style="font-size: 0.9rem; color: #6c757d; background-color: rgba(0,0,0,0.05); padding: 0.25rem 0.5rem; border-radius: 4px;">
                                Secondary Model
                            </div>
                        </div>
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                            <div style="font-size: 2.5rem;">{rf_icon}</div>
                            <div>
                                <h2 style="margin: 0; color: {rf_color};">{rf_class}</h2>
                                <p style="margin: 0; color: #6c757d;">with {rf_confidence:.1%} confidence</p>
                            </div>
                        </div>
                        <div class="confidence-meter">
                            <div class="fill" style="width: {rf_confidence*100}%; background: {get_prediction_color(rf_confidence)}"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # LIME Explanation (moved up so top_words is available)
                with st.spinner("Generating explanation..."):
                    explainer = LimeTextExplainer(class_names=["Fake", "Real"])
                    exp = explainer.explain_instance(
                        cleaned_text_input,
                        lambda x: lr_model.predict_proba(vectorizer.transform(x)),
                        num_features=10
                    )
                    # Convert explanation to DataFrame
                    exp_data = pd.DataFrame(
                        exp.as_list(),
                        columns=['Word/Phrase', 'Impact']
                    ).sort_values('Impact', key=abs, ascending=False)
                    top_words = exp_data.head(3)

                # Model Agreement with improved styling
                st.markdown("""
                <div style="margin: 3rem 0 1.5rem 0;">
                    <h2 style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: #5BC0BE;">🤝</span> Model Consensus
                    </h2>
                </div>
                """, unsafe_allow_html=True)

                # Calculate average confidence
                avg_confidence = (lr_confidence + rf_confidence) / 2

                if lr_pred == rf_pred:
                    # Models agree - show a nice card
                    agreement_color = "#2E7D32" if lr_pred == 1 else "#C62828"
                    agreement_bg = "rgba(46, 125, 50, 0.05)" if lr_pred == 1 else "rgba(198, 40, 40, 0.05)"
                    agreement_icon = "✅" if lr_pred == 1 else "❌"

                    st.markdown(f"""
                    <div style="background: {agreement_bg}; border-left: 5px solid {agreement_color};
                                padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; animation: fadeIn 1s ease-in-out;">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="font-size: 2.5rem;">{agreement_icon}</div>
                            <div>
                                <h3 style="margin: 0; color: {agreement_color};">Strong Consensus: This article appears to be {lr_class}</h3>
                                <p style="margin: 0.5rem 0 0 0; color: #6c757d;">
                                    Both models agree on the classification with an average confidence of {avg_confidence:.1%}
                                </p>
                            </div>
                        </div>
                        <div style="margin-top: 1rem;">
                            <div class="confidence-meter" style="height: 12px;">
                                <div class="fill" style="width: {avg_confidence*100}%; background: {get_prediction_color(avg_confidence)}"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Models disagree - show warning card
                    st.markdown(f"""
                    <div style="background: rgba(255, 209, 102, 0.1); border-left: 5px solid #FFD166;
                                padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; animation: fadeIn 1s ease-in-out;">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="font-size: 2.5rem;">⚠️</div>
                            <div>
                                <h3 style="margin: 0; color: #b7791f;">Models Disagree on Classification</h3>
                                <p style="margin: 0.5rem 0 0 0; color: #6c757d;">
                                    Our models have different opinions about this article
                                </p>
                            </div>
                        </div>
                        <div style="margin-top: 1rem; background: rgba(255, 255, 255, 0.5); padding: 1rem; border-radius: 8px;">
                            <p style="margin: 0 0 0.5rem 0; font-weight: 500; color: #6c757d;">This could mean:</p>
                            <ul style="margin: 0; padding-left: 1.5rem; color: #6c757d;">
                                <li>The article has mixed characteristics of both real and fake news</li>
                                <li>The content is ambiguous or contains both factual and misleading elements</li>
                                <li>More detailed analysis might be needed to determine authenticity</li>
                            </ul>
                            <p style="margin: 1rem 0 0 0; font-style: italic; color: #6c757d;">
                                Consider the confidence scores of each model and use the LLM verification feature below for additional analysis.
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # EXPLANATION SECTION with improved styling
                st.markdown("""
                <div style="margin: 3rem 0 1.5rem 0;">
                    <h2 style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: #5BC0BE;">🧠</span> Why This Classification?
                    </h2>
                </div>
                """, unsafe_allow_html=True)

                # Explanation card based on prediction
                if lr_pred == 0:
                    # Fake news explanation
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(198, 40, 40, 0.03), rgba(198, 40, 40, 0.07));
                                border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                                margin-bottom: 2rem; animation: fadeIn 1s ease-in-out;">
                        <div style="padding: 1.5rem;">
                            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                <div style="width: 50px; height: 50px; border-radius: 50%; background-color: #C62828;
                                            display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                                    ❌
                                </div>
                                <div>
                                    <h3 style="margin: 0; color: #C62828;">Why This Appears to be Fake News</h3>
                                    <p style="margin: 0.25rem 0 0 0; color: #6c757d;">
                                        Our model detected several patterns commonly associated with misleading content
                                    </p>
                                </div>
                            </div>

                            <div style="margin-top: 1.5rem;">
                                <div style="margin-bottom: 1rem;">
                                    <div style="font-weight: 600; color: #3A506B; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                                        <span style="color: #C62828;">🔍</span> Key Influential Words/Phrases
                                    </div>
                                    <div style="background-color: rgba(255, 255, 255, 0.7); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #C62828;">
                                        <span style="color: #C62828; font-weight: 500;">{', '.join(top_words['Word/Phrase'])}</span>
                                    </div>
                                </div>

                                <div style="margin-bottom: 1rem;">
                                    <div style="font-weight: 600; color: #3A506B; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                                        <span style="color: #C62828;">📝</span> Writing Style Indicators
                                    </div>
                                    <div style="background-color: rgba(255, 255, 255, 0.7); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #C62828;">
                                        <ul style="margin: 0; padding-left: 1.5rem; color: #6c757d;">
                                            <li>Excessive use of exclamation points or question marks</li>
                                            <li>Unusual capitalization patterns (ALL CAPS for emphasis)</li>
                                            <li>Sensationalist language and emotional appeals</li>
                                            <li>Overly simplistic explanations for complex topics</li>
                                        </ul>
                                    </div>
                                </div>

                                <div>
                                    <div style="font-weight: 600; color: #3A506B; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                                        <span style="color: #C62828;">😮</span> Sentiment Analysis
                                    </div>
                                    <div style="background-color: rgba(255, 255, 255, 0.7); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #C62828;">
                                        <ul style="margin: 0; padding-left: 1.5rem; color: #6c757d;">
                                            <li>Unusually strong emotional polarity (very positive or very negative)</li>
                                            <li>High subjectivity score indicating opinion rather than fact</li>
                                            <li>Emotional language designed to provoke strong reactions</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Real news explanation
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, rgba(46, 125, 50, 0.03), rgba(46, 125, 50, 0.07));
                                border-radius: 10px; overflow: hidden; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                                margin-bottom: 2rem; animation: fadeIn 1s ease-in-out;">
                        <div style="padding: 1.5rem;">
                            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                <div style="width: 50px; height: 50px; border-radius: 50%; background-color: #2E7D32;
                                            display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                                    ✅
                                </div>
                                <div>
                                    <h3 style="margin: 0; color: #2E7D32;">Why This Appears to be Real News</h3>
                                    <p style="margin: 0.25rem 0 0 0; color: #6c757d;">
                                        Our model identified several patterns typically found in credible journalism
                                    </p>
                                </div>
                            </div>

                            <div style="margin-top: 1.5rem;">
                                <div style="margin-bottom: 1rem;">
                                    <div style="font-weight: 600; color: #3A506B; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                                        <span style="color: #2E7D32;">🔍</span> Key Influential Words/Phrases
                                    </div>
                                    <div style="background-color: rgba(255, 255, 255, 0.7); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #2E7D32;">
                                        <span style="color: #2E7D32; font-weight: 500;">{', '.join(top_words['Word/Phrase'])}</span>
                                    </div>
                                </div>

                                <div style="margin-bottom: 1rem;">
                                    <div style="font-weight: 600; color: #3A506B; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                                        <span style="color: #2E7D32;">📝</span> Writing Style Indicators
                                    </div>
                                    <div style="background-color: rgba(255, 255, 255, 0.7); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #2E7D32;">
                                        <ul style="margin: 0; padding-left: 1.5rem; color: #6c757d;">
                                            <li>Balanced use of punctuation and proper capitalization</li>
                                            <li>Structured paragraphs with clear sentence flow</li>
                                            <li>Factual language with appropriate attribution</li>
                                            <li>Nuanced explanations that acknowledge complexity</li>
                                        </ul>
                                    </div>
                                </div>

                                <div>
                                    <div style="font-weight: 600; color: #3A506B; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                                        <span style="color: #2E7D32;">😊</span> Sentiment Analysis
                                    </div>
                                    <div style="background-color: rgba(255, 255, 255, 0.7); padding: 0.75rem; border-radius: 8px; border-left: 3px solid #2E7D32;">
                                        <ul style="margin: 0; padding-left: 1.5rem; color: #6c757d;">
                                            <li>Neutral or moderate emotional tone</li>
                                            <li>Lower subjectivity score indicating fact-based reporting</li>
                                            <li>Balanced presentation of information without emotional manipulation</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Enhanced LLM Factual Accuracy Check (if Ollama is available)
                if LLM_AVAILABLE and ollama_status:
                    st.markdown("""
                    <div style="margin: 3rem 0 1.5rem 0;">
                        <h2 style="display: flex; align-items: center; gap: 0.5rem;">
                            <span style="color: #5BC0BE;">🤖</span> Enhanced Factual Accuracy Check
                        </h2>
                        <p style="color: #6c757d; max-width: 800px; margin: 0.5rem 0 1.5rem 0;">
                            Our AI assistant will check for factual errors by searching the internet for verification
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Create columns for model selection
                    col1, col2 = st.columns(2)

                    with col1:
                        # Model selection
                        model_options = ollama_status["models"] if "models" in ollama_status else ["deepseek-r1:7b", "mistral"]
                        default_model = "deepseek-r1:7b" if "deepseek-r1:7b" in model_options else model_options[0] if model_options else "deepseek-r1:7b"

                        selected_model = st.selectbox(
                            "Select LLM Model",
                            options=model_options if model_options else ["mistral"],
                            index=model_options.index(default_model) if default_model in model_options and model_options else 0,
                            help="Choose which language model to use for verification"
                        )

                    with col2:
                        # Temperature setting
                        temperature = st.slider(
                            "Temperature",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.2,
                            step=0.1,
                            help="Lower values make the output more deterministic (recommended: 0.2)"
                        )

                    # Verify button
                    verify_button = st.button(
                        "🔍 Perform Factual Accuracy Check",
                        type="primary",
                        help="Use LLM and internet search to verify facts in this article"
                    )

                    if verify_button:
                        # Progress bar
                        progress_bar = st.progress(0)

                        with st.spinner("Performing enhanced factual accuracy check..."):
                            try:
                                # Initialize the verifier
                                verifier = load_llm_verifier(model_name=selected_model, temperature=temperature)

                                if verifier:
                                    # Update progress
                                    progress_bar.progress(20)
                                    st.markdown("""
                                    <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.5rem;">
                                        <span style="color: #5BC0BE;">⏳</span> Extracting entities and claims from article...
                                    </div>
                                    """, unsafe_allow_html=True)

                                    # Check factual accuracy with the improved method
                                    factual_check = verifier.check_factual_accuracy(cleaned_text_input)

                                    # Update progress
                                    progress_bar.progress(60)
                                    st.markdown("""
                                    <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.5rem;">
                                        <span style="color: #5BC0BE;">⏳</span> Analyzing search results and verifying facts...
                                    </div>
                                    """, unsafe_allow_html=True)

                                    time.sleep(1)  # Small delay for UI feedback
                                    progress_bar.progress(100)

                                    # Display the results with percentage
                                    # Get the real and fake percentages if available, otherwise calculate from contains_errors
                                    real_percentage = factual_check.get("real_percentage", 0 if factual_check["contains_errors"] else 100)
                                    fake_percentage = factual_check.get("fake_percentage", 100 if factual_check["contains_errors"] else 0)

                                    # Determine color based on percentage
                                    if real_percentage >= 75:  # Strongly real
                                        verdict_color = "#2E7D32"  # Green
                                        verdict_icon = "✅"
                                        verdict_text = "Highly Credible Content"
                                    elif real_percentage >= 60:  # Moderately real
                                        verdict_color = "#66BB6A"  # Light green
                                        verdict_icon = "✅"
                                        verdict_text = "Mostly Credible Content"
                                    elif real_percentage >= 40:  # Mixed/Uncertain
                                        verdict_color = "#FFD166"  # Yellow
                                        verdict_icon = "⚠️"
                                        verdict_text = "Mixed Credibility"
                                    elif real_percentage >= 25:  # Moderately fake
                                        verdict_color = "#FB8C00"  # Orange
                                        verdict_icon = "⚠️"
                                        verdict_text = "Questionable Content"
                                    else:  # Strongly fake
                                        verdict_color = "#C62828"  # Red
                                        verdict_icon = "❌"
                                        verdict_text = "Factual Errors Detected"

                                    st.markdown(f"""
                                        <div style="background: linear-gradient(135deg, rgba({198 if real_percentage < 50 else 46}, {40 if real_percentage < 50 else 125}, {40 if real_percentage < 50 else 50}, 0.05), rgba(255, 255, 255, 0.5));
                                                    padding: 1.5rem; border-radius: 10px; margin: 1.5rem 0; animation: fadeIn 1s ease-in-out;
                                                    border-left: 5px solid {verdict_color};">
                                            <div style="display: flex; align-items: center; gap: 1rem;">
                                                <div style="font-size: 2.5rem;">{verdict_icon}</div>
                                                <div>
                                                    <h3 style="margin: 0; color: {verdict_color};">{verdict_text}</h3>
                                                    <div style="margin: 0.75rem 0 0.5rem 0;">
                                                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                                            <span style="color: #C62828; font-weight: 600;">Fake: {fake_percentage}%</span>
                                                            <span style="color: #2E7D32; font-weight: 600;">Real: {real_percentage}%</span>
                                                        </div>
                                                        <div style="height: 8px; background-color: #f0f0f0; border-radius: 4px; overflow: hidden;">
                                                            <div style="width: {real_percentage}%; height: 100%; background: linear-gradient(to right, #C62828, #FFD166, #2E7D32); border-radius: 4px;"></div>
                                                        </div>
                                                    </div>
                                                    <p style="margin: 0.5rem 0 0 0; color: #6c757d;">
                                                        Confidence: {factual_check["confidence"]}
                                                    </p>
                                                </div>
                                            </div>

                                            <div style="margin-top: 1.5rem;">
                                                <h4 style="color: #3A506B; margin-bottom: 0.75rem;">Entities Analyzed</h4>
                                                <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem;">
                                        """, unsafe_allow_html=True)

                                    # Display entities as tags (up to 10)
                                    entity_color = "rgba(198, 40, 40, 0.1)" if factual_check["contains_errors"] else "rgba(46, 125, 50, 0.1)"
                                    entity_border = "rgba(198, 40, 40, 0.2)" if factual_check["contains_errors"] else "rgba(46, 125, 50, 0.2)"
                                    entities_to_show = factual_check.get("entities", [])[:10] if "entities" in factual_check else []

                                    for entity in entities_to_show:
                                        st.markdown(f"""
                                        <span style="background-color: {entity_color};
                                                    padding: 0.3rem 0.6rem;
                                                    border-radius: 20px;
                                                    font-size: 0.85rem;
                                                    color: #3A506B;
                                                    border: 1px solid {entity_border};">
                                            {entity}
                                        </span>
                                        """, unsafe_allow_html=True)

                                    st.markdown("""
                                            </div>
                                        """, unsafe_allow_html=True)

                                    # Analysis section
                                    analysis_color = "#C62828" if factual_check["contains_errors"] else "#2E7D32"
                                    st.markdown(f"""
                                            <div style="margin-top: 1rem; background: rgba(255, 255, 255, 0.7); padding: 1.5rem; border-radius: 8px;">
                                                <h4 style="margin-top: 0; color: {analysis_color};">Analysis:</h4>
                                                <div style="color: #3A506B; font-size: 0.95rem; line-height: 1.5;">
                                                    {factual_check["analysis"]}
                                                </div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)

                                    # Override the model prediction if we found factual errors and confidence is not low
                                    if factual_check["contains_errors"] and factual_check["confidence"] != "Low":
                                        st.markdown(f"""
                                        <div style="background: rgba(198, 40, 40, 0.05); border-left: 3px solid #C62828; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                                            <p style="margin: 0; color: #6c757d;">
                                                <span style="font-weight: 600; color: #C62828;">Important:</span> Based on the factual errors detected,
                                                this article should be classified as <b>Fake News</b> regardless of the model prediction.
                                            </p>
                                        </div>
                                        """, unsafe_allow_html=True)

                                    # Show verification results in expander
                                    verification_results = factual_check.get("verification_results", [])
                                    if verification_results:
                                        with st.expander("View Verification Details", expanded=False):
                                            for i, result in enumerate(verification_results, 1):
                                                border_color = "#C62828" if factual_check["contains_errors"] else "#2E7D32"
                                                st.markdown(f"""
                                                <div style="margin-bottom: 1.5rem; padding: 1rem; background-color: white; border-radius: 8px; border-left: 3px solid {border_color};">
                                                    <h4 style="margin-top: 0; color: #3A506B;">Entity/Claim {i}: {result["entity"]}</h4>
                                                    <div style="background-color: #f8f9fa; padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem; font-size: 0.9rem; max-height: 200px; overflow-y: auto;">
                                                        <pre style="white-space: pre-wrap; margin: 0; color: #3A506B;">{result["search_results"]}</pre>
                                                    </div>
                                                </div>
                                                """, unsafe_allow_html=True)
                                else:
                                    st.error("Failed to initialize the LLM verifier. Please check if Ollama is running.")
                            except Exception as e:
                                st.error(f"Error performing factual check: {str(e)}")
                                st.info("Please make sure Ollama is running and the selected model is available.")
                    else:
                        # Show a placeholder with instructions when button is not clicked
                        st.markdown("""
                        <div style="background-color: rgba(91, 192, 190, 0.1); padding: 1.5rem; border-radius: 8px; text-align: center; margin: 1rem 0;">
                            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                            <h3 style="margin-top: 0; color: #3A506B; font-size: 1.2rem;">Click the button above to verify facts</h3>
                            <p style="color: #6c757d; margin-bottom: 0;">
                                The AI will search the internet to verify key facts and entities in the article
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                # Feature Analysis with improved styling
                st.markdown("""
                <div style="margin: 3rem 0 1.5rem 0;">
                    <h2 style="display: flex; align-items: center; gap: 0.5rem;">
                        <span style="color: #5BC0BE;">🔍</span> Detailed Feature Analysis
                    </h2>
                    <p style="color: #6c757d; max-width: 800px; margin: 0.5rem 0 1.5rem 0;">
                        These are the specific features extracted from the text that influenced the model's decision
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Tabs for different types of analysis
                tab1, tab2, tab3 = st.tabs(["📊 Word Impact", "📝 Stylometric Features", "😊 Sentiment Analysis"])

                with tab1:
                    st.markdown("""
                    <div style="margin: 1rem 0;">
                        <h3 style="font-size: 1.3rem; color: #3A506B; margin-bottom: 1rem;">
                            Words and Phrases Influencing the Prediction
                        </h3>
                        <p style="color: #6c757d; margin-bottom: 1.5rem;">
                            The chart below shows which words and phrases had the most impact on the model's decision.
                            Blue bars indicate words that suggest real news, while red bars suggest fake news.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Improved Plotly chart
                    fig = go.Figure(go.Bar(
                        x=exp_data['Impact'],
                        y=exp_data['Word/Phrase'],
                        orientation='h',
                        marker_color=['#C62828' if x < 0 else '#2E7D32' for x in exp_data['Impact']],
                        marker_line_width=1,
                        marker_line_color='white',
                        hoverinfo='text',
                        hovertext=[f"{word}: {impact:.3f} impact" for word, impact in zip(exp_data['Word/Phrase'], exp_data['Impact'])]
                    ))

                    fig.update_layout(
                        title=None,
                        xaxis_title="Impact (negative = suggests fake, positive = suggests real)",
                        yaxis_title=None,
                        height=400,
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(
                            gridcolor='rgba(0,0,0,0.1)',
                            zerolinecolor='rgba(0,0,0,0.2)'
                        ),
                        yaxis=dict(
                            gridcolor='rgba(0,0,0,0)'
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Explanation box
                    if lr_pred == 0:
                        reason = ', '.join([f"'{word}'" for word in top_words['Word/Phrase']])
                        st.markdown(f"""
                        <div style="background-color: rgba(198, 40, 40, 0.05); border-left: 3px solid #C62828; padding: 1rem; border-radius: 8px;">
                            <p style="margin: 0; color: #6c757d;">
                                <span style="font-weight: 600; color: #C62828;">Key Finding:</span> This article was classified as <b>Fake</b> because words/phrases like {reason} contributed most to the prediction.
                                These terms are often associated with sensationalist or misleading content.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        reason = ', '.join([f"'{word}'" for word in top_words['Word/Phrase']])
                        st.markdown(f"""
                        <div style="background-color: rgba(46, 125, 50, 0.05); border-left: 3px solid #2E7D32; padding: 1rem; border-radius: 8px;">
                            <p style="margin: 0; color: #6c757d;">
                                <span style="font-weight: 600; color: #2E7D32;">Key Finding:</span> This article was classified as <b>Real</b> because words/phrases like {reason} contributed most to the prediction.
                                These terms are commonly found in factual reporting and credible journalism.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                with tab2:
                    st.markdown("""
                    <div style="margin: 1rem 0;">
                        <h3 style="font-size: 1.3rem; color: #3A506B; margin-bottom: 1rem;">
                            Writing Style Analysis
                        </h3>
                        <p style="color: #6c757d; margin-bottom: 1.5rem;">
                            These metrics analyze the writing style of the article, including sentence structure, punctuation usage, and capitalization patterns.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Create a more visually appealing display of stylometric features
                    style_df = pd.DataFrame([style_features])

                    # Create two columns for the features
                    col1, col2 = st.columns(2)

                    with col1:
                        # Text length and sentence metrics
                        st.markdown("""
                        <div style="background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); height: 100%;">
                            <h4 style="color: #3A506B; margin-top: 0; margin-bottom: 1rem; font-size: 1.1rem;">Text Structure</h4>
                        """, unsafe_allow_html=True)

                        # Text length
                        text_length = style_features['text_length']
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                <div style="color: #6c757d; font-size: 0.9rem;">Text Length</div>
                                <div style="font-weight: 600; color: #3A506B;">{text_length:,} characters</div>
                            </div>
                            <div style="height: 6px; background-color: #f0f0f0; border-radius: 3px; overflow: hidden;">
                                <div style="height: 100%; width: {min(text_length/5000*100, 100)}%; background-color: #5BC0BE;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Average sentence length
                        avg_sent_len = style_features['avg_sentence_length']
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                <div style="color: #6c757d; font-size: 0.9rem;">Avg. Sentence Length</div>
                                <div style="font-weight: 600; color: #3A506B;">{avg_sent_len:.1f} words</div>
                            </div>
                            <div style="height: 6px; background-color: #f0f0f0; border-radius: 3px; overflow: hidden;">
                                <div style="height: 100%; width: {min(avg_sent_len/30*100, 100)}%; background-color: #5BC0BE;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                    with col2:
                        # Punctuation and capitalization metrics
                        st.markdown("""
                        <div style="background-color: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); height: 100%;">
                            <h4 style="color: #3A506B; margin-top: 0; margin-bottom: 1rem; font-size: 1.1rem;">Punctuation & Capitalization</h4>
                        """, unsafe_allow_html=True)

                        # Exclamation density
                        excl_density = style_features['exclamation_density'] * 100
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                <div style="color: #6c757d; font-size: 0.9rem;">Exclamation Marks</div>
                                <div style="font-weight: 600; color: #3A506B;">{excl_density:.2f}%</div>
                            </div>
                            <div style="height: 6px; background-color: #f0f0f0; border-radius: 3px; overflow: hidden;">
                                <div style="height: 100%; width: {min(excl_density*100, 100)}%; background-color: #5BC0BE;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Question density
                        q_density = style_features['question_density'] * 100
                        st.markdown(f"""
                        <div style="margin-bottom: 1rem;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                <div style="color: #6c757d; font-size: 0.9rem;">Question Marks</div>
                                <div style="font-weight: 600; color: #3A506B;">{q_density:.2f}%</div>
                            </div>
                            <div style="height: 6px; background-color: #f0f0f0; border-radius: 3px; overflow: hidden;">
                                <div style="height: 100%; width: {min(q_density*100, 100)}%; background-color: #5BC0BE;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Caps ratio
                        caps_ratio = style_features['caps_ratio'] * 100
                        st.markdown(f"""
                        <div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                <div style="color: #6c757d; font-size: 0.9rem;">ALL CAPS Words</div>
                                <div style="font-weight: 600; color: #3A506B;">{caps_ratio:.2f}%</div>
                            </div>
                            <div style="height: 6px; background-color: #f0f0f0; border-radius: 3px; overflow: hidden;">
                                <div style="height: 100%; width: {min(caps_ratio*10, 100)}%; background-color: #5BC0BE;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("</div>", unsafe_allow_html=True)

                    # Interpretation
                    st.markdown("""
                    <div style="background-color: rgba(91, 192, 190, 0.1); padding: 1rem; border-radius: 8px; margin-top: 1.5rem;">
                        <h4 style="color: #3A506B; margin-top: 0; margin-bottom: 0.5rem; font-size: 1rem;">Interpretation</h4>
                        <p style="color: #6c757d; margin: 0; font-size: 0.9rem;">
                            Fake news often contains more exclamation points, question marks, and ALL CAPS words to create emotional impact.
                            Real news typically has more balanced punctuation usage and follows standard capitalization rules.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with tab3:
                    st.markdown("""
                    <div style="margin: 1rem 0;">
                        <h3 style="font-size: 1.3rem; color: #3A506B; margin-bottom: 1rem;">
                            Emotional Tone Analysis
                        </h3>
                        <p style="color: #6c757d; margin-bottom: 1.5rem;">
                            These metrics analyze the emotional tone and subjectivity of the article, which can help distinguish between factual reporting and opinion-based content.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Create a more visually appealing display of sentiment features
                    sentiment_df = pd.DataFrame([sentiment_features])

                    # Create a gauge chart for polarity
                    polarity = sentiment_features['polarity']
                    polarity_color = "#2E7D32" if polarity > 0 else "#C62828" if polarity < 0 else "#3A506B"

                    fig_polarity = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = polarity,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Sentiment Polarity", 'font': {'color': "#3A506B", 'size': 16}},
                        gauge = {
                            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#3A506B"},
                            'bar': {'color': polarity_color},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [-1, -0.5], 'color': 'rgba(198, 40, 40, 0.2)'},
                                {'range': [-0.5, 0], 'color': 'rgba(198, 40, 40, 0.1)'},
                                {'range': [0, 0.5], 'color': 'rgba(46, 125, 50, 0.1)'},
                                {'range': [0.5, 1], 'color': 'rgba(46, 125, 50, 0.2)'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 2},
                                'thickness': 0.75,
                                'value': polarity
                            }
                        }
                    ))

                    fig_polarity.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#3A506B", 'family': "Inter"}
                    )

                    # Create a gauge chart for subjectivity
                    subjectivity = sentiment_features['subjectivity']

                    fig_subjectivity = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = subjectivity,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Subjectivity", 'font': {'color': "#3A506B", 'size': 16}},
                        gauge = {
                            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#3A506B"},
                            'bar': {'color': "#5BC0BE"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 0.33], 'color': 'rgba(91, 192, 190, 0.1)'},
                                {'range': [0.33, 0.66], 'color': 'rgba(91, 192, 190, 0.2)'},
                                {'range': [0.66, 1], 'color': 'rgba(91, 192, 190, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 2},
                                'thickness': 0.75,
                                'value': subjectivity
                            }
                        }
                    ))

                    fig_subjectivity.update_layout(
                        height=250,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': "#3A506B", 'family': "Inter"}
                    )

                    # Display the charts side by side
                    col1, col2 = st.columns(2)

                    with col1:
                        st.plotly_chart(fig_polarity, use_container_width=True)
                        st.markdown("""
                        <div style="text-align: center; margin-top: -1rem;">
                            <p style="color: #6c757d; font-size: 0.9rem;">
                                <span style="color: #C62828;">Negative (-1)</span> to
                                <span style="color: #2E7D32;">Positive (+1)</span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.plotly_chart(fig_subjectivity, use_container_width=True)
                        st.markdown("""
                        <div style="text-align: center; margin-top: -1rem;">
                            <p style="color: #6c757d; font-size: 0.9rem;">
                                <span style="color: #3A506B;">Objective (0)</span> to
                                <span style="color: #5BC0BE;">Subjective (1)</span>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Interpretation
                    st.markdown("""
                    <div style="background-color: rgba(91, 192, 190, 0.1); padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                        <h4 style="color: #3A506B; margin-top: 0; margin-bottom: 0.5rem; font-size: 1rem;">Interpretation</h4>
                        <p style="color: #6c757d; margin: 0; font-size: 0.9rem;">
                            <strong>Polarity</strong> measures the emotional tone from negative (-1) to positive (+1). Fake news often has extreme polarity (very negative or very positive).
                            <br><br>
                            <strong>Subjectivity</strong> measures how opinion-based (1) versus fact-based (0) the content is. Fake news typically has higher subjectivity scores.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)



            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.info("Please try again with different text or contact support.")

    else:
        st.info("Please enter some text to analyze.")

        # Example section with improved styling
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

                <div style="margin-top: 1rem;">
                    <button onclick="
                        const textarea = document.querySelector('textarea');
                        if (textarea) {
                            textarea.value = 'Scientists have discovered a new species of butterfly in the Amazon rainforest. The species, named Morpho azure, displays an unprecedented shade of blue in its wings. The research team spent three years studying the butterfly\\'s habitat and behavior. Their findings were published in the Journal of Lepidopterology.';
                            const event = new Event('input', { bubbles: true });
                            textarea.dispatchEvent(event);
                        }
                    "
                    style="background-color: #5BC0BE; color: white; border: none; padding: 0.5rem 1rem; border-radius: 4px; cursor: pointer; font-size: 0.9rem;">
                        Use This Example
                    </button>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.error("Unable to load models. Please check if all required files are present in the models directory.")

# Information about LLM verification with improved styling
with st.expander("About LLM Verification with Internet Search", expanded=False):
    st.markdown("""
    <div style="padding: 1rem; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
        <h3 style="color: #3A506B; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #5BC0BE;">🤖</span> How LLM Verification Works
        </h3>

        <p style="color: #6c757d;">
            This feature uses a locally running Large Language Model (LLM) combined with web search to verify news articles:
        </p>

        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1.5rem 0;">
            <div style="background: linear-gradient(135deg, rgba(91, 192, 190, 0.1), rgba(255, 255, 255, 1));
                        border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem;">
                    <div style="background-color: #5BC0BE; color: white; width: 24px; height: 24px; border-radius: 50%;
                                display: flex; align-items: center; justify-content: center; font-size: 0.8rem;">1</div>
                    <h4 style="margin: 0; color: #3A506B;">Key Claim Extraction</h4>
                </div>
                <p style="color: #6c757d; margin: 0; padding-left: 2rem;">
                    The LLM identifies the main factual claims in the article
                </p>
            </div>

            <div style="background: linear-gradient(135deg, rgba(91, 192, 190, 0.1), rgba(255, 255, 255, 1));
                        border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem;">
                    <div style="background-color: #5BC0BE; color: white; width: 24px; height: 24px; border-radius: 50%;
                                display: flex; align-items: center; justify-content: center; font-size: 0.8rem;">2</div>
                    <h4 style="margin: 0; color: #3A506B;">Web Search</h4>
                </div>
                <p style="color: #6c757d; margin: 0; padding-left: 2rem;">
                    Each claim is searched on the web to find supporting or contradicting information
                </p>
            </div>

            <div style="background: linear-gradient(135deg, rgba(91, 192, 190, 0.1), rgba(255, 255, 255, 1));
                        border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem;">
                    <div style="background-color: #5BC0BE; color: white; width: 24px; height: 24px; border-radius: 50%;
                                display: flex; align-items: center; justify-content: center; font-size: 0.8rem;">3</div>
                    <h4 style="margin: 0; color: #3A506B;">Analysis</h4>
                </div>
                <p style="color: #6c757d; margin: 0; padding-left: 2rem;">
                    The LLM analyzes the search results and compares them with the article
                </p>
            </div>

            <div style="background: linear-gradient(135deg, rgba(91, 192, 190, 0.1), rgba(255, 255, 255, 1));
                        border-radius: 8px; padding: 1rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                <div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem;">
                    <div style="background-color: #5BC0BE; color: white; width: 24px; height: 24px; border-radius: 50%;
                                display: flex; align-items: center; justify-content: center; font-size: 0.8rem;">4</div>
                    <h4 style="margin: 0; color: #3A506B;">Verdict</h4>
                </div>
                <p style="color: #6c757d; margin: 0; padding-left: 2rem;">
                    Based on the analysis, the LLM provides a verdict on whether the article appears to be real or fake
                </p>
            </div>
        </div>

        <div style="display: flex; gap: 1.5rem; margin-top: 2rem;">
            <div style="flex: 1;">
                <h3 style="color: #3A506B; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #5BC0BE;">🔒</span> Privacy & Performance
                </h3>
                <ul style="color: #6c757d; padding-left: 1.5rem;">
                    <li>All LLM processing happens locally on your machine</li>
                    <li>No article data is sent to external AI services</li>
                    <li>Performance depends on your hardware and the model size</li>
                    <li>First-time model loading may take a few moments</li>
                </ul>
            </div>

            <div style="flex: 1;">
                <h3 style="color: #3A506B; display: flex; align-items: center; gap: 0.5rem;">
                    <span style="color: #5BC0BE;">⚠️</span> Limitations
                </h3>
                <ul style="color: #6c757d; padding-left: 1.5rem;">
                    <li>The system is not perfect and should be used as one tool among many for fact-checking</li>
                    <li>Results depend on the quality of web search results</li>
                    <li>The LLM may occasionally hallucinate or make errors in its analysis</li>
                    <li>Always verify important information with multiple trusted sources</li>
                </ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)