import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Basic page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        font-size: 3rem !important;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    """Load the preprocessed news dataset"""
    try:
        # Get the absolute path to the CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(os.path.dirname(current_dir), "news_with_sentiment.csv")
        
        if not os.path.exists(csv_path):
            st.error(f"File not found at {csv_path}")
            return None
            
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.write("Current working directory:", os.getcwd())
        return None

def show_overview(df):
    st.title("🔍 Fake News Detection System")
    st.markdown("### A Machine Learning Approach to Identify Fake News")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Articles", len(df))
    with col2:
        fake_count = len(df[df['label'] == 'fake'])
        st.metric("Fake News Articles", fake_count)
    with col3:
        real_count = len(df[df['label'] == 'real'])
        st.metric("Real News Articles", real_count)

def show_sentiment_analysis(df):
    st.header("📊 Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Articles")
        fig = px.pie(df, names='label', title='Real vs Fake News Distribution',
                    color_discrete_map={'real': '#2ecc71', 'fake': '#e74c3c'})
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Average Sentiment by Category")
        avg_sentiment = df.groupby('label')['polarity'].mean().reset_index()
        fig = px.bar(avg_sentiment, x='label', y='polarity',
                    title='Average Sentiment Polarity by News Category',
                    color='label', color_discrete_map={'real': '#2ecc71', 'fake': '#e74c3c'})
        st.plotly_chart(fig, use_container_width=True)

def show_stylometric_analysis(df):
    st.header("📝 Stylometric Analysis")
    
    # Calculate average stylometric features by label
    style_cols = ['exclamation_density', 'question_density', 'quotes_density', 
                  'capitalized_ratio', 'all_caps_ratio', 'special_chars_density']
    
    style_means = df.groupby('label')[style_cols].mean().reset_index()
    style_means_long = pd.melt(style_means, id_vars=['label'], 
                              value_vars=style_cols,
                              var_name='Feature', value_name='Value')
    
    # Create visualization
    fig = px.bar(style_means_long, x='Feature', y='Value', color='label',
                 barmode='group', title='Stylometric Features by Article Type',
                 color_discrete_map={'real': '#2ecc71', 'fake': '#e74c3c'})
    
    fig.update_layout(
        xaxis_title="Feature Type",
        yaxis_title="Average Density/Ratio",
        xaxis={'categoryorder':'total descending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_sample_articles(df):
    st.header("📰 Sample Articles")
    label_filter = st.radio("Select article type:", ["All", "Real", "Fake"])
    
    if label_filter == "All":
        sample_df = df
    else:
        sample_df = df[df['label'].str.lower() == label_filter.lower()]
    
    num_samples = st.slider("Number of samples to show", 1, 5, 3)
    samples = sample_df.sample(n=min(num_samples, len(sample_df)))
    
    for _, row in samples.iterrows():
        with st.expander(f"{row['label'].upper()} News - {row['title'][:100]}..."):
            st.markdown(f"**Title:** {row['title']}")
            st.markdown("**Text Preview:**")
            st.markdown(f"_{row['text'][:200]}..._")
            
            # Create two columns for metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Sentiment Metrics:**")
                st.markdown(f"- Polarity: {row['polarity']:.2f}")
                st.markdown(f"- Subjectivity: {row['subjectivity']:.2f}")
            with col2:
                st.markdown("**Stylometric Metrics:**")
                st.markdown(f"- Exclamation Density: {row['exclamation_density']:.3f}")
                st.markdown(f"- Question Density: {row['question_density']:.3f}")
                st.markdown(f"- Quotes Density: {row['quotes_density']:.3f}")

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
        ["Overview", "Sentiment Analysis", "Stylometric Analysis", "Sample Articles"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("""
        This application uses machine learning to detect fake news articles. 
        It analyzes both content and writing style to make predictions.
        
        Features analyzed include:
        - Sentiment Analysis
        - Stylometric Features
        - Text Statistics
    """)
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Display selected page
        if page == "Overview":
            show_overview(df)
            show_sentiment_analysis(df)
        elif page == "Sentiment Analysis":
            show_sentiment_analysis(df)
        elif page == "Stylometric Analysis":
            show_stylometric_analysis(df)
        elif page == "Sample Articles":
            show_sample_articles(df)

if __name__ == "__main__":
    main()
