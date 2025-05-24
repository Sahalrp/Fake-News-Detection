import streamlit as st
import sys
import os
import json
import pandas as pd
import plotly.express as px

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer_model import TransformerModel
from src.explainability.explainer import ModelExplainer

# Model configuration
MODEL_CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3
}

@st.cache_resource
def load_model():
    """Load the pre-trained model."""
    model = TransformerModel(MODEL_CONFIG)
    model_path = os.path.join('models', 'fake_news_bert')
    
    if os.path.exists(model_path):
        model.load(model_path)
    else:
        st.warning("""⚠️ No pre-trained model found. The system will use the base BERT model without fine-tuning.
        This is a demo mode and predictions may not be accurate. Please train the model first.""")
        # Initialize the model with base BERT
        model.initialize_model()
    
    return model

@st.cache_resource
def get_explainer(_model):
    """Create model explainer."""
    return ModelExplainer(
        predict_fn=_model.predict_proba,
        tokenizer=_model.tokenizer
    )

def main():
    st.set_page_config(
        page_title="Fake News Detection System",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Fake News Detection System")
    st.markdown("""
    This system uses advanced Natural Language Processing and Machine Learning techniques
    to classify news articles as real or fake, with explanations for its decisions.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        try:
            model = load_model()
            explainer = get_explainer(model)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    # Input section
    st.header("📝 Input Text")
    text_input = st.text_area(
        "Enter news article text:",
        height=200,
        help="Paste the news article text you want to analyze"
    )
    
    analyze_button = st.button("Analyze Text")
    
    if analyze_button and text_input:
        with st.spinner("Analyzing..."):
            try:
                # Get prediction
                probabilities = model.predict_proba([text_input])[0]
                prediction = "FAKE" if probabilities[1] > probabilities[0] else "REAL"
                confidence = float(max(probabilities))
                
                # Get explanations
                lime_explanation = explainer.explain_with_lime(text_input)
                
                # Display results
                st.header("🎯 Results")
                
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction")
                    prediction_color = "red" if prediction == "FAKE" else "green"
                    st.markdown(f"""
                    <div style='padding: 20px; border-radius: 5px; background-color: {prediction_color}25;'>
                        <h1 style='text-align: center; color: {prediction_color};'>{prediction}</h1>
                        <p style='text-align: center;'>Confidence: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability distribution
                    prob_df = pd.DataFrame({
                        'Class': ['Real', 'Fake'],
                        'Probability': probabilities
                    })
                    fig = px.bar(
                        prob_df,
                        x='Class',
                        y='Probability',
                        color='Class',
                        color_discrete_map={'Real': 'green', 'Fake': 'red'}
                    )
                    st.plotly_chart(fig)
                
                with col2:
                    st.subheader("Explanation")
                    # Display feature importance
                    feature_importance = pd.DataFrame(
                        lime_explanation['feature_importance'].items(),
                        columns=['Feature', 'Importance']
                    )
                    feature_importance['Abs_Importance'] = abs(feature_importance['Importance'])
                    feature_importance = feature_importance.sort_values(
                        'Abs_Importance',
                        ascending=True
                    )
                    
                    fig = px.bar(
                        feature_importance,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig)
                
                # Display highlighted text
                st.subheader("📑 Text Analysis")
                st.markdown(lime_explanation['explanation_html'], unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
    
    # Add information about the system
    st.sidebar.title("ℹ️ About")
    st.sidebar.markdown("""
    This system uses state-of-the-art Natural Language Processing and Machine Learning
    techniques to detect fake news. It employs:
    
    - **BERT**: Pre-trained transformer model fine-tuned for fake news detection
    - **LIME**: Local Interpretable Model-agnostic Explanations
    - **SHAP**: SHapley Additive exPlanations
    
    The system provides both predictions and explanations to help users understand
    why a particular piece of text was classified as real or fake.
    """)
    
    # Add model details
    st.sidebar.title("🤖 Model Details")
    st.sidebar.markdown(f"""
    - **Base Model**: {MODEL_CONFIG['model_name']}
    - **Max Length**: {MODEL_CONFIG['max_length']} tokens
    - **Batch Size**: {MODEL_CONFIG['batch_size']}
    """)

if __name__ == "__main__":
    main() 