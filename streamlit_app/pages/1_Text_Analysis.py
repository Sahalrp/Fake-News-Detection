import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data
from shared_styles import load_css
from shared_components import hide_sidebar_items

# Page config
st.set_page_config(page_title="Text Analysis", page_icon="📊", layout="wide")

# Load shared CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Hide Model Explanation and Prediction from sidebar
hide_sidebar_items()

# Page title with animation
st.markdown("""
<div class="page-title">
    <h1>📊 Text Analysis Dashboard</h1>
    <p>Explore stylometric and sentiment patterns in news articles to understand what differentiates real from fake news</p>
</div>
""", unsafe_allow_html=True)

# Get root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Debug information in expander
with st.expander("Debug Information", expanded=False):
    st.write("Current working directory:", os.getcwd())
    st.write("Project root directory:", ROOT_DIR)
    st.write("Models directory:", os.path.join(ROOT_DIR, "models"))

# Using load_data from utils.py

# Load data with a spinner
with st.spinner("Loading dataset..."):
    df = load_data(ROOT_DIR)

if df is not None:
    # Dataset Overview with improved styling
    st.markdown("""
    <h2 style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="color: #5BC0BE;">📈</span> Dataset Overview
    </h2>
    """, unsafe_allow_html=True)

    # Introduction card
    st.markdown("""
    <div class="slide-in" style="background: linear-gradient(135deg, rgba(58, 80, 107, 0.05), rgba(91, 192, 190, 0.05));
                padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;">
        <p style="color: #3A506B; margin-bottom: 0;">
            This dashboard provides insights into the textual patterns found in our news dataset.
            By analyzing these patterns, we can identify characteristics that differentiate real from fake news.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Metrics in cards with hover effects
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem; font-weight: 700; color: #3A506B; margin-bottom: 0.5rem;">{len(df):,}</div>
            <div style="font-size: 1.1rem; color: #6c757d; font-weight: 500;">Total Articles</div>
            <div style="width: 50px; height: 4px; background: #5BC0BE; margin-top: 1rem;"></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        real_percent = (df['label'] == 'Real').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem; font-weight: 700; color: #2E7D32; margin-bottom: 0.5rem;">{real_percent:.1f}%</div>
            <div style="font-size: 1.1rem; color: #6c757d; font-weight: 500;">Real News</div>
            <div style="width: 50px; height: 4px; background: #2E7D32; margin-top: 1rem;"></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        fake_percent = (df['label'] == 'Fake').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem; font-weight: 700; color: #C62828; margin-bottom: 0.5rem;">{fake_percent:.1f}%</div>
            <div style="font-size: 1.1rem; color: #6c757d; font-weight: 500;">Fake News</div>
            <div style="width: 50px; height: 4px; background: #C62828; margin-top: 1rem;"></div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_length = df['text_length'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 0.5rem;">{avg_length:.0f}</div>
            <div style="font-size: 1.1rem; color: #6c757d; font-weight: 500;">Avg. Length (chars)</div>
            <div style="width: 50px; height: 4px; background: #1E88E5; margin-top: 1rem;"></div>
        </div>
        """, unsafe_allow_html=True)

    # Stylometric Analysis with improved styling
    st.markdown("""
    <h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 3rem;">
        <span style="color: #5BC0BE;">📝</span> Stylometric Analysis
    </h2>
    <div style="margin-bottom: 1.5rem;">
        <p style="color: #6c757d;">
            Stylometric analysis examines the writing style characteristics that may differ between real and fake news articles.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Text Length Distribution with improved styling
    st.markdown("""
    <h3 style="display: flex; align-items: center; gap: 0.5rem;">
        <span style="color: #3A506B;">📏</span> Text Length Analysis
    </h3>
    """, unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:
        # Enhanced histogram with better colors
        fig_length = px.histogram(
            df,
            x='text_length',
            color='label',
            title='Text Length Distribution',
            labels={'text_length': 'Number of Characters', 'count': 'Number of Articles'},
            marginal='box',
            color_discrete_map={'Real': '#2E7D32', 'Fake': '#C62828'},
            opacity=0.8
        )

        fig_length.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(
                family="Inter, sans-serif",
                size=12,
                color="#3A506B"
            )
        )

        st.plotly_chart(fig_length, use_container_width=True)

    with col6:
        # Enhanced violin plot with better colors
        fig_words = px.violin(
            df,
            y='avg_sentence_length',
            color='label',
            box=True,
            title='Average Sentence Length by Category',
            labels={'avg_sentence_length': 'Words per Sentence'},
            color_discrete_map={'Real': '#2E7D32', 'Fake': '#C62828'},
        )

        fig_words.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(
                family="Inter, sans-serif",
                size=12,
                color="#3A506B"
            )
        )

        st.plotly_chart(fig_words, use_container_width=True)

    # Punctuation Analysis with improved styling
    st.markdown("""
    <h3 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 2rem;">
        <span style="color: #3A506B;">🔤</span> Punctuation Patterns
    </h3>
    <div style="margin-bottom: 1.5rem;">
        <p style="color: #6c757d;">
            Punctuation usage can reveal important stylistic differences between real and fake news articles.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Card container for the charts
    st.markdown("""
    <div style="background: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 2rem;">
    """, unsafe_allow_html=True)

    col7, col8, col9 = st.columns(3)

    with col7:
        fig_excl = px.box(
            df,
            y='exclamation_density',
            color='label',
            title='Exclamation Mark Usage',
            points='all',
            color_discrete_map={'Real': '#2E7D32', 'Fake': '#C62828'}
        )

        fig_excl.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(family="Inter, sans-serif", size=14),
            font=dict(family="Inter, sans-serif", size=12, color="#3A506B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_excl, use_container_width=True)

    with col8:
        fig_quest = px.box(
            df,
            y='question_density',
            color='label',
            title='Question Mark Usage',
            points='all',
            color_discrete_map={'Real': '#2E7D32', 'Fake': '#C62828'}
        )

        fig_quest.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(family="Inter, sans-serif", size=14),
            font=dict(family="Inter, sans-serif", size=12, color="#3A506B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_quest, use_container_width=True)

    with col9:
        fig_quotes = px.box(
            df,
            y='quotes_density',
            color='label',
            title='Quotation Mark Usage',
            points='all',
            color_discrete_map={'Real': '#2E7D32', 'Fake': '#C62828'}
        )

        fig_quotes.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(family="Inter, sans-serif", size=14),
            font=dict(family="Inter, sans-serif", size=12, color="#3A506B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_quotes, use_container_width=True)

    # Close the card container
    st.markdown("</div>", unsafe_allow_html=True)

    # Add insight box
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(91, 192, 190, 0.1), rgba(58, 80, 107, 0.1));
                border-left: 4px solid #5BC0BE; padding: 1rem; border-radius: 0 8px 8px 0; margin: 1.5rem 0;">
        <h4 style="margin-top: 0; color: #3A506B; font-weight: 600;">Key Insight</h4>
        <p style="margin-bottom: 0; color: #3A506B;">
            Fake news articles often use more exclamation marks and question marks to evoke emotional responses,
            while real news tends to use more quotation marks to cite sources and provide evidence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sentiment Analysis with improved styling
    st.markdown("""
    <h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 3rem;">
        <span style="color: #5BC0BE;">😊</span> Sentiment Analysis
    </h2>
    <div style="margin-bottom: 1.5rem;">
        <p style="color: #6c757d;">
            Sentiment analysis examines the emotional tone and subjectivity of news articles, which can help identify potential bias.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Card container for sentiment charts
    st.markdown("""
    <div style="background: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 2rem;">
    """, unsafe_allow_html=True)

    col10, col11 = st.columns(2)

    with col10:
        fig_pol = px.histogram(
            df,
            x='sentiment_polarity',
            color='label',
            title='Sentiment Polarity Distribution',
            labels={'sentiment_polarity': 'Polarity (-1 to 1)', 'count': 'Number of Articles'},
            marginal='violin',
            color_discrete_map={'Real': '#2E7D32', 'Fake': '#C62828'},
            opacity=0.8,
            barmode='overlay'
        )

        fig_pol.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(family="Inter, sans-serif", size=16),
            font=dict(family="Inter, sans-serif", size=12, color="#3A506B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_pol, use_container_width=True)

    with col11:
        fig_subj = px.histogram(
            df,
            x='sentiment_subjectivity',
            color='label',
            title='Subjectivity Distribution',
            labels={'sentiment_subjectivity': 'Subjectivity (0 to 1)', 'count': 'Number of Articles'},
            marginal='violin',
            color_discrete_map={'Real': '#2E7D32', 'Fake': '#C62828'},
            opacity=0.8,
            barmode='overlay'
        )

        fig_subj.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(family="Inter, sans-serif", size=16),
            font=dict(family="Inter, sans-serif", size=12, color="#3A506B"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig_subj, use_container_width=True)

    # Close the card container
    st.markdown("</div>", unsafe_allow_html=True)

    # Add sentiment explanation box
    st.markdown("""
    <div class="slide-in" style="display: flex; gap: 1.5rem; margin: 2rem 0;">
        <div style="flex: 1; background: linear-gradient(135deg, rgba(46, 125, 50, 0.1), rgba(255, 255, 255, 1));
                    border-radius: 8px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
            <h4 style="color: #2E7D32; margin-top: 0;">Polarity</h4>
            <p style="color: #3A506B; margin-bottom: 0;">
                Polarity measures how positive or negative the text is, ranging from -1 (very negative) to 1 (very positive).
                Real news tends to be more neutral, while fake news may lean toward extremes.
            </p>
        </div>
        <div style="flex: 1; background: linear-gradient(135deg, rgba(198, 40, 40, 0.1), rgba(255, 255, 255, 1));
                    border-radius: 8px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
            <h4 style="color: #C62828; margin-top: 0;">Subjectivity</h4>
            <p style="color: #3A506B; margin-bottom: 0;">
                Subjectivity measures how opinionated the text is, ranging from 0 (objective) to 1 (subjective).
                Fake news often contains more subjective language to influence readers' opinions.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Correlation Analysis with improved styling
    st.markdown("""
    <h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 3rem;">
        <span style="color: #5BC0BE;">🔄</span> Feature Correlations
    </h2>
    <div style="margin-bottom: 1.5rem;">
        <p style="color: #6c757d;">
            This correlation matrix shows relationships between different text features, helping identify patterns that distinguish real from fake news.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Select numerical columns
    numerical_cols = [
        'text_length', 'avg_sentence_length', 'exclamation_density',
        'question_density', 'quotes_density', 'caps_ratio',
        'sentiment_polarity', 'sentiment_subjectivity'
    ]

    correlation_matrix = df[numerical_cols].corr()

    # Enhanced correlation heatmap
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=numerical_cols,
        y=numerical_cols,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))

    fig_corr.update_layout(
        title={
            'text': 'Feature Correlation Matrix',
            'font': {'family': 'Inter, sans-serif', 'size': 20, 'color': '#3A506B'}
        },
        height=600,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", size=12, color="#3A506B")
    )

    # Card container for correlation matrix
    st.markdown("""
    <div style="background: white; border-radius: 10px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 2rem;">
    """, unsafe_allow_html=True)

    st.plotly_chart(fig_corr, use_container_width=True)

    # Close the card container
    st.markdown("</div>", unsafe_allow_html=True)

    # Statistical Summary with improved styling
    st.markdown("""
    <h2 style="display: flex; align-items: center; gap: 0.5rem; margin-top: 3rem;">
        <span style="color: #5BC0BE;">📊</span> Statistical Summary
    </h2>
    <div style="margin-bottom: 1.5rem;">
        <p style="color: #6c757d;">
            Detailed statistical breakdown of text features by category (real vs. fake news).
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Enhanced expander for statistics
    with st.expander("View Detailed Statistics", expanded=False):
        # Calculate statistics by category
        stats_df = df.groupby('label')[numerical_cols].agg(['mean', 'std', 'min', 'max']).round(3)

        # Add styling to the dataframe
        st.markdown("""
        <div style="background: white; border-radius: 8px; padding: 1.5rem; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
        """, unsafe_allow_html=True)

        st.write(stats_df)

        st.markdown("""
        <div style="margin-top: 1rem; font-size: 0.9rem; color: #6c757d;">
            <strong>Note:</strong> These statistics show the mean, standard deviation, minimum, and maximum values for each feature,
            separated by real and fake news categories.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Add footer
    st.markdown("""
    <div style="margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #e0e0e0; text-align: center;">
        <p style="color: #6c757d; font-size: 0.9rem;">
            This analysis is based on a dataset of news articles collected and labeled for research purposes.
            The patterns identified here may vary in different contexts or with different datasets.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("Unable to load the dataset. Please check the file path and try again.")