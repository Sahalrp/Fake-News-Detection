import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lime.lime_text import LimeTextExplainer
import shap
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_models
from shared_styles import load_css

# Page configuration
st.set_page_config(page_title="Model Explanation", page_icon="🔍", layout="wide")

# Load shared CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Additional custom CSS for this page
st.markdown("""
<style>
    /* Ensure all text is visible with proper contrast */
    p, span, div, li, a, label, text {
        color: #333333 !important;
    }
    
    .loading {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #5BC0BE;
        animation: spin 1s ease-in-out infinite;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Analysis card styling */
    .analysis-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }

    /* Prediction result styling */
    .prediction-result {
        display: flex;
        align-items: center;
        gap: 1rem;
        background: linear-gradient(135deg, rgba(58, 80, 107, 0.1), rgba(91, 192, 190, 0.1));
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .prediction-icon {
        font-size: 2.5rem;
    }

    .prediction-text {
        flex-grow: 1;
    }

    .prediction-text h3 {
        margin: 0 0 0.5rem 0;
        color: #3A506B;
    }

    .prediction-text p {
        margin: 0;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Page title with animation
st.markdown("""
<div class="page-title">
    <h1>🔍 Model Explanation Dashboard</h1>
    <p>Understanding how our AI model makes predictions using Explainable AI techniques</p>
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
                This dashboard uses explainable AI techniques to help you understand how our model analyzes news articles.
                Enter a news article below to see which words and phrases influence the model's prediction.
            </p>
            <p style="color: #6c757d; margin-bottom: 0;">
                For best results, enter a complete news article with at least several paragraphs.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Get the absolute path to the project root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Function to load only the needed models for this page
def load_explanation_models():
    # Use the shared load_models function but only return what we need
    lr_model, _, vectorizer = load_models(ROOT_DIR)
    return lr_model, vectorizer

# Show loading state while loading models
with st.spinner("Loading models..."):
    model, vectorizer = load_explanation_models()

# Input validation
def validate_input(text):
    if not text:
        return False, "Please enter some text to analyze."
    if len(text.split()) < 10:
        return False, "Please enter a longer text (at least 10 words) for better analysis."
    if len(text) > 10000:
        return False, "Text is too long. Please enter a shorter text (maximum 10000 characters)."
    return True, ""

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
    "Enter news text for analysis:",
    height=200,
    help="Enter the news article text you want to analyze. For best results, provide at least a few sentences.",
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

# Validate input and show appropriate messages
is_valid, error_message = validate_input(text_input)

if text_input and (analyze_button or 'analyze_clicked' in st.session_state):
    # Store that analysis was clicked
    if analyze_button:
        st.session_state.analyze_clicked = True

    if not is_valid:
        st.error(error_message)
    elif not model or not vectorizer:
        st.error("Models not loaded correctly. Please check the error messages above.")
    else:
        try:
            with st.spinner("Analyzing text..."):
                # Get prediction and probability
                X_vec = vectorizer.transform([text_input])
                prediction = model.predict(X_vec)[0]
                probabilities = model.predict_proba(X_vec)[0]
                confidence = max(probabilities) * 100

                # Display prediction with enhanced styling
                prediction_class = "real" if prediction == 1 else "fake"
                confidence_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"

                st.markdown(f"""
                <div class="prediction-result">
                    <div class="prediction-icon">
                        {"✅" if prediction == 1 else "❌"}
                    </div>
                    <div class="prediction-text">
                        <h3>Prediction: {"Real" if prediction == 1 else "Fake"} News</h3>
                        <p>Confidence: {confidence:.1f}% ({confidence_level})</p>
                    </div>
                    <div style="width: 100px; height: 100px; position: relative;">
                        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border-radius: 50%;
                                    background: conic-gradient({"#2E7D32" if prediction == 1 else "#C62828"} {confidence}%, #e0e0e0 0%);"></div>
                        <div style="position: absolute; top: 10px; left: 10px; width: 80px; height: 80px; border-radius: 50%; background: white;
                                    display: flex; align-items: center; justify-content: center; font-weight: bold; color: #3A506B;">
                            {confidence:.0f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Progress bar for analysis sections
                progress_bar = st.progress(0)

                # Section header with improved styling
                st.markdown("""
                <h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 2rem;">
                    <span style="color: #5BC0BE;">🔤</span> Word Importance Analysis (LIME)
                </h2>
                <div style="margin-bottom: 1.5rem;">
                    <p style="color: #6c757d;">
                        LIME (Local Interpretable Model-agnostic Explanations) shows which words and phrases most influenced the prediction.
                    </p>
                </div>
                """, unsafe_allow_html=True)

                progress_bar.progress(25)

                # Card container for LIME analysis
                st.markdown("""
                <div class="analysis-card">
                """, unsafe_allow_html=True)

                with st.spinner("Generating LIME explanation..."):
                    explainer = LimeTextExplainer(class_names=["Fake", "Real"])
                    exp = explainer.explain_instance(
                        text_input,
                        lambda x: model.predict_proba(vectorizer.transform(x)),
                        num_features=10
                    )

                    # Convert LIME explanation to DataFrame
                    word_importance = pd.DataFrame(
                        exp.as_list(),
                        columns=['Word/Phrase', 'Impact']
                    ).sort_values('Impact')

                    # Create enhanced word importance plot
                    fig_lime = go.Figure(go.Bar(
                        x=word_importance['Impact'],
                        y=word_importance['Word/Phrase'],
                        orientation='h',
                        marker_color=word_importance['Impact'].apply(
                            lambda x: '#C62828' if x < 0 else '#2E7D32'
                        ),
                        text=word_importance['Impact'].apply(lambda x: f"{abs(x):.3f}"),
                        textposition='auto',
                        hoverinfo='text',
                        hovertext=word_importance.apply(
                            lambda row: f"Word/Phrase: {row['Word/Phrase']}<br>Impact: {row['Impact']:.3f}<br>Suggests: {'Fake' if row['Impact'] < 0 else 'Real'} News",
                            axis=1
                        )
                    ))

                    fig_lime.update_layout(
                        title={
                            'text': "Word/Phrase Impact on Prediction",
                            'font': {'family': 'Inter, sans-serif', 'size': 18, 'color': '#3A506B'}
                        },
                        xaxis_title="Impact (negative = fake, positive = real)",
                        xaxis=dict(
                            zeroline=True,
                            zerolinewidth=2,
                            zerolinecolor='#e0e0e0',
                            gridcolor='#f5f5f5'
                        ),
                        yaxis=dict(
                            gridcolor='#f5f5f5'
                        ),
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Inter, sans-serif", size=12, color="#3A506B"),
                        margin=dict(l=10, r=10, t=50, b=10)
                    )

                    st.plotly_chart(fig_lime, use_container_width=True)

                    # Add explanation text
                    st.markdown("""
                    <div style="margin-top: 1rem; padding: 1rem; background-color: rgba(91, 192, 190, 0.1); border-radius: 8px;">
                        <h4 style="margin-top: 0; color: #3A506B;">How to Interpret This Chart</h4>
                        <ul style="color: #6c757d; margin-bottom: 0;">
                            <li><span style="color: #2E7D32; font-weight: 500;">Green bars</span> indicate words/phrases that suggest <strong>real news</strong></li>
                            <li><span style="color: #C62828; font-weight: 500;">Red bars</span> indicate words/phrases that suggest <strong>fake news</strong></li>
                            <li>Longer bars indicate stronger influence on the prediction</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # Close the card container
                st.markdown("</div>", unsafe_allow_html=True)

                progress_bar.progress(50)

                # Feature Importance and SHAP Analysis with improved layout
                st.markdown("""
                <div style="display: flex; gap: 2rem; margin: 2rem 0;">
                """, unsafe_allow_html=True)

                col3, col4 = st.columns(2)

                with col3:
                    # Global Feature Importance with improved styling
                    st.markdown("""
                    <h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0;">
                        <span style="color: #5BC0BE;">🌐</span> Global Feature Importance
                    </h2>
                    <div style="margin-bottom: 1rem;">
                        <p style="color: #6c757d; font-size: 0.9rem;">
                            These are the words that generally have the most influence on predictions across all articles.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Card container for feature importance
                    st.markdown("""
                    <div class="analysis-card">
                    """, unsafe_allow_html=True)

                    with st.spinner("Calculating feature importance..."):
                        feature_names = vectorizer.get_feature_names_out()
                        coefficients = model.coef_[0]

                        top_n = 10
                        top_indices = np.argsort(np.abs(coefficients))[-top_n:]
                        top_features = feature_names[top_indices]
                        top_coefficients = coefficients[top_indices]

                        # Enhanced feature importance plot
                        fig_importance = go.Figure(go.Bar(
                            x=top_coefficients,
                            y=top_features,
                            orientation='h',
                            marker_color=['#C62828' if x < 0 else '#2E7D32' for x in top_coefficients],
                            text=np.abs(top_coefficients).round(3),
                            textposition='auto',
                            hoverinfo='text',
                            hovertext=[f"Word: {word}<br>Coefficient: {coef:.3f}<br>Suggests: {'Fake' if coef < 0 else 'Real'} News"
                                      for word, coef in zip(top_features, top_coefficients)]
                        ))

                        fig_importance.update_layout(
                            title={
                                'text': f"Top {top_n} Most Important Words",
                                'font': {'family': 'Inter, sans-serif', 'size': 16, 'color': '#3A506B'}
                            },
                            xaxis_title="Coefficient Value",
                            xaxis=dict(
                                zeroline=True,
                                zerolinewidth=2,
                                zerolinecolor='#e0e0e0',
                                gridcolor='#f5f5f5'
                            ),
                            yaxis=dict(
                                gridcolor='#f5f5f5'
                            ),
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter, sans-serif", size=12, color="#3A506B"),
                            margin=dict(l=10, r=10, t=50, b=10)
                        )

                        st.plotly_chart(fig_importance, use_container_width=True)

                    # Close the card container
                    st.markdown("</div>", unsafe_allow_html=True)

                progress_bar.progress(75)

                with col4:
                    # SHAP Analysis with improved styling
                    st.markdown("""
                    <h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0;">
                        <span style="color: #5BC0BE;">🧩</span> SHAP Analysis
                    </h2>
                    <div style="margin-bottom: 1rem;">
                        <p style="color: #6c757d; font-size: 0.9rem;">
                            SHAP values show how each word contributed to this specific prediction.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Card container for SHAP analysis
                    st.markdown("""
                    <div class="analysis-card">
                    """, unsafe_allow_html=True)

                    with st.spinner("Calculating SHAP values..."):
                        explainer = shap.LinearExplainer(model, vectorizer.transform([""]))
                        shap_values = explainer.shap_values(X_vec)

                        feature_names = vectorizer.get_feature_names_out()
                        shap_df = pd.DataFrame({
                            'feature': feature_names,
                            'shap_value': shap_values[0]
                        })
                        shap_df = shap_df[shap_df['shap_value'] != 0].sort_values('shap_value')

                        # Take top 10 by absolute value if there are too many
                        if len(shap_df) > 10:
                            top_indices = np.argsort(np.abs(shap_df['shap_value']))[-10:]
                            shap_df = shap_df.iloc[top_indices]

                        # Enhanced SHAP plot
                        fig_shap = go.Figure(go.Bar(
                            x=shap_df['shap_value'],
                            y=shap_df['feature'],
                            orientation='h',
                            marker_color=shap_df['shap_value'].apply(
                                lambda x: '#C62828' if x < 0 else '#2E7D32'
                            ),
                            text=shap_df['shap_value'].apply(lambda x: f"{abs(x):.3f}"),
                            textposition='auto',
                            hoverinfo='text',
                            hovertext=shap_df.apply(
                                lambda row: f"Word: {row['feature']}<br>SHAP value: {row['shap_value']:.3f}<br>Suggests: {'Fake' if row['shap_value'] < 0 else 'Real'} News",
                                axis=1
                            )
                        ))

                        fig_shap.update_layout(
                            title={
                                'text': "SHAP Values for Current Prediction",
                                'font': {'family': 'Inter, sans-serif', 'size': 16, 'color': '#3A506B'}
                            },
                            xaxis_title="SHAP Value",
                            xaxis=dict(
                                zeroline=True,
                                zerolinewidth=2,
                                zerolinecolor='#e0e0e0',
                                gridcolor='#f5f5f5'
                            ),
                            yaxis=dict(
                                gridcolor='#f5f5f5'
                            ),
                            height=400,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family="Inter, sans-serif", size=12, color="#3A506B"),
                            margin=dict(l=10, r=10, t=50, b=10)
                        )

                        st.plotly_chart(fig_shap, use_container_width=True)

                    # Close the card container
                    st.markdown("</div>", unsafe_allow_html=True)

                # Close the flex container
                st.markdown("</div>", unsafe_allow_html=True)

                progress_bar.progress(100)

                # Explanation Summary with improved styling
                st.markdown("""
                <h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 3rem;">
                    <span style="color: #5BC0BE;">📋</span> Summary
                </h2>
                """, unsafe_allow_html=True)

                # Summary card with enhanced styling
                st.markdown("""
                <div class="analysis-card">
                    <h3 style="margin-top: 0; color: #3A506B;">Understanding the Results</h3>

                    <p style="color: #6c757d; margin-bottom: 1.5rem;">
                        This analysis shows how the model arrived at its prediction. Here's what each section tells you:
                    </p>

                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem;">
                        <div style="background: linear-gradient(135deg, rgba(91, 192, 190, 0.1), rgba(255, 255, 255, 1));
                                    border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                            <h4 style="margin-top: 0; color: #3A506B; display: flex; align-items: center; gap: 0.5rem;">
                                <span style="color: #5BC0BE;">1</span> Confidence Score
                            </h4>
                            <p style="color: #6c757d; margin-bottom: 0;">
                                Shows how certain the model is about its prediction. Higher confidence means the model is more sure about its classification.
                            </p>
                        </div>

                        <div style="background: linear-gradient(135deg, rgba(91, 192, 190, 0.1), rgba(255, 255, 255, 1));
                                    border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                            <h4 style="margin-top: 0; color: #3A506B; display: flex; align-items: center; gap: 0.5rem;">
                                <span style="color: #5BC0BE;">2</span> Word Importance
                            </h4>
                            <p style="color: #6c757d; margin-bottom: 0;">
                                Shows which specific words/phrases influenced the prediction.
                                <span style="color: #2E7D32; font-weight: 500;">Green bars</span> suggest real news, while
                                <span style="color: #C62828; font-weight: 500;">red bars</span> suggest fake news.
                            </p>
                        </div>

                        <div style="background: linear-gradient(135deg, rgba(91, 192, 190, 0.1), rgba(255, 255, 255, 1));
                                    border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                            <h4 style="margin-top: 0; color: #3A506B; display: flex; align-items: center; gap: 0.5rem;">
                                <span style="color: #5BC0BE;">3</span> Global Feature Importance
                            </h4>
                            <p style="color: #6c757d; margin-bottom: 0;">
                                Shows which words are generally important for all predictions, based on the model's learned coefficients.
                            </p>
                        </div>

                        <div style="background: linear-gradient(135deg, rgba(91, 192, 190, 0.1), rgba(255, 255, 255, 1));
                                    border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
                            <h4 style="margin-top: 0; color: #3A506B; display: flex; align-items: center; gap: 0.5rem;">
                                <span style="color: #5BC0BE;">4</span> SHAP Values
                            </h4>
                            <p style="color: #6c757d; margin-bottom: 0;">
                                Shows how each word contributed to this specific prediction, taking into account the unique context of your article.
                            </p>
                        </div>
                    </div>

                    <div style="margin-top: 2rem; padding: 1rem; background-color: rgba(58, 80, 107, 0.1); border-radius: 8px;">
                        <p style="color: #3A506B; margin-bottom: 0; font-style: italic;">
                            <strong>Note:</strong> The longer and more detailed the text you provide, the more accurate the analysis will be.
                            For best results, use complete news articles with multiple paragraphs.
                        </p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Add footer
                st.markdown("""
                <div style="margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #e0e0e0; text-align: center;">
                    <p style="color: #6c757d; font-size: 0.9rem;">
                        This explanation dashboard uses LIME and SHAP, state-of-the-art techniques in Explainable AI,
                        to help you understand how machine learning models make decisions.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.info("Please try again with different text or contact support if the problem persists.")
else:
    st.info("Please enter some text to analyze. The text should be a news article or similar content.")