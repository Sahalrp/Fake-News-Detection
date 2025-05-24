import streamlit as st
from streamlit import switch_page
st.set_page_config(
    page_title="Fake News Detection - Home",
    page_icon="📰",
    layout="wide"
)

import pandas as pd
import plotly.express as px
import os
from utils import load_data, load_models, check_system_status, check_ollama_status
from html_renderer import render_html, render_styled_html, render_status_indicator, render_button_group
from shared_components import hide_sidebar_items

# Hide Model Explanation and Prediction from sidebar
hide_sidebar_items()

# Define a modern color palette
# Primary: #3A506B (dark blue)
# Secondary: #5BC0BE (teal)
# Accent: #FF6B6B (coral)
# Background: #F5F5F5 (light gray)
# Dark: #1C2541 (navy)

# Comprehensive custom CSS for a modern, cohesive design
st.markdown("""
    <style>
    /* Global styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #333333 !important;
    }
    
    /* Ensure all text is visible with proper contrast */
    p, span, div, li, a, label, text {
        color: #333333 !important;
    }

    /* Main content area styling */
    .main {
        background-color: #F5F5F5;
        padding: 1rem;
    }

    .main .block-container {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }

    /* Sidebar styling - enhanced for consistency across all pages */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1C2541 0%, #3A506B 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1) !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Ensure all sidebar text is visible */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: white !important;
        opacity: 0.9 !important;
    }

    /* Simple approach to rename "app" to "Home" in the sidebar */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:first-child a p {
        visibility: hidden;
        position: relative;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:first-child a p:after {
        content: 'Home';
        visibility: visible;
        position: absolute;
        left: 0;
        top: 0;
    }

    /* Hide Model Explanation and Prediction from sidebar - more direct approach */
    /* Target by position - 3rd and 4th items in the sidebar navigation */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:nth-child(3),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:nth-child(4) {
        display: none !important;
    }

    /* Target by text content */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a p {
        visibility: visible;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a[href*="Model_Explanation"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li a[href*="Prediction"] {
        display: none !important;
    }

    /* Target parent elements */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li:has(a[href*="Model_Explanation"]),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] ul li:has(a[href*="Prediction"]) {
        display: none !important;
    }

    /* Brute force approach */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href="/Model_Explanation"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href="/Prediction"] {
        display: none !important;
    }

    /* Target by text content directly */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:has(p:contains("Model Explanation")),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:has(p:contains("Prediction")) {
        display: none !important;
    }

    /* Target by file name in the sidebar */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="2_Model_Explanation"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="3_Prediction"] {
        display: none !important;
    }

    /* Target the parent li elements of these links */
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:has(a[href*="2_Model_Explanation"]),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:has(a[href*="3_Prediction"]) {
        display: none !important;
    }

    /* More targeted approach for sidebar items */
    section[data-testid="stSidebar"] *:has(text="Model Explanation"),
    section[data-testid="stSidebar"] *:has(text="Prediction"),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:nth-of-type(3),
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:nth-of-type(4) {
        display: none !important;
    }

    /* JavaScript approach will be used as backup */

    /* Make sidebar headers stand out */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: white !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] hr {
        margin: 1.5rem 0 !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Style sidebar buttons */
    section[data-testid="stSidebar"] button {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: white !important;
        transition: all 0.2s ease !important;
    }

    section[data-testid="stSidebar"] button:hover {
        background-color: rgba(255, 255, 255, 0.2) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Style sidebar selectbox and other inputs */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Ensure sidebar navigation is consistent */
    section[data-testid="stSidebar"] a {
        color: #5BC0BE !important;
        text-decoration: none !important;
        transition: color 0.2s ease !important;
    }

    section[data-testid="stSidebar"] a:hover {
        color: white !important;
        text-decoration: underline !important;
    }

    /* Header styling */
    h1 {
        color: #1C2541;
        font-weight: 700;
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #5BC0BE;
    }

    h2 {
        color: #3A506B;
        font-weight: 600;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        color: #3A506B;
        font-weight: 500;
        font-size: 1.4rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Button styling */
    .stButton > button {
        background-color: #5BC0BE !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
    }

    .stButton > button:hover {
        background-color: #3A506B !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Primary button */
    .stButton > button[data-baseweb="button"][kind="primary"] {
        background-color: #FF6B6B !important;
    }

    .stButton > button[data-baseweb="button"][kind="primary"]:hover {
        background-color: #e05e5e !important;
    }
    
    /* Quick Links and Resources buttons */
    .stButton > button[key*="btn"] {
        background-color: #3A506B !important;
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton > button[key*="btn"]:hover {
        background-color: #1C2541 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #3A506B !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 500 !important;
        color: #6c757d !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }

    /* Card styling */
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-indicator.success {
        background-color: #5BC0BE;
        box-shadow: 0 0 8px rgba(91, 192, 190, 0.5);
    }

    .status-indicator.error {
        background-color: #FF6B6B;
        box-shadow: 0 0 8px rgba(255, 107, 107, 0.5);
    }

    .status-indicator.warning {
        background-color: #FFD166;
        box-shadow: 0 0 8px rgba(255, 209, 102, 0.5);
    }

    /* Main header styling */
    .main-header, .main-title-text {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1C2541 !important;
        margin: 0 !important;
        padding: 0 !important;
        text-align: center;
        /* Remove gradient text that causes visibility issues */
        /*background: linear-gradient(90deg, #3A506B, #5BC0BE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;*/
        line-height: 1.2 !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* Hero section styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(28, 37, 65, 0.8), rgba(58, 80, 107, 0.8)), url('https://images.unsplash.com/photo-1504711434969-e33886168f5c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80');
        background-size: cover;
        background-position: center;
        border-radius: 10px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .hero-title {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        color: #5BC0BE !important;
        /* Remove gradient text that causes visibility issues */
        /*background: linear-gradient(90deg, #5BC0BE, white);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;*/
        display: inline-block;
        border-bottom: none !important;
        padding-bottom: 0 !important;
    }

    .hero-subtitle {
        font-size: 1.5rem !important;
        font-weight: 400 !important;
        margin-bottom: 2rem !important;
        color: rgba(255, 255, 255, 0.9);
        max-width: 800px;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Alert styling */
    .st-success {
        background-color: rgba(91, 192, 190, 0.2) !important;
        border: 1px solid #5BC0BE !important;
        color: #2c7c7a !important;
    }

    .st-error {
        background-color: rgba(255, 107, 107, 0.2) !important;
        border: 1px solid #FF6B6B !important;
        color: #c53030 !important;
    }

    .st-warning {
        background-color: rgba(255, 209, 102, 0.2) !important;
        border: 1px solid #FFD166 !important;
        color: #b7791f !important;
    }

    .st-info {
        background-color: rgba(58, 80, 107, 0.2) !important;
        border: 1px solid #3A506B !important;
        color: #2d3748 !important;
    }

    /* Expander styling - improved for better UX */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #3A506B !important;
        background-color: rgba(91, 192, 190, 0.1) !important;
        border-radius: 4px !important;
        padding: 0.75rem 1rem !important;
        transition: background-color 0.2s ease !important;
    }

    .streamlit-expanderHeader:hover {
        background-color: rgba(91, 192, 190, 0.2) !important;
    }

    .streamlit-expanderContent {
        border-left: 1px solid #5BC0BE !important;
        padding-left: 1rem !important;
        margin-left: 0.5rem !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* Make expanders more compact by default */
    .streamlit-expander {
        margin-bottom: 1rem !important;
    }

    /* Separator styling */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #5BC0BE, transparent);
    }

    /* Footer styling */
    footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        color: #6c757d;
        font-size: 0.9rem;
    }

    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1C2541;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }

    /* Code block styling - completely hide by default */
    .stCodeBlock, div[data-testid="stCodeBlock"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
    }

    /* Ensure the title and banner containers are always visible */
    div.element-container:has(div.main-title-container),
    div.element-container:has(h1.main-title-text),
    div.element-container:has(div[style*="background: linear-gradient(90deg, #5BC0BE, #3A506B, #5BC0BE"]) {
        display: block !important;
        visibility: visible !important;
        height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 1 !important;
        position: relative !important;
        z-index: 1000 !important;
    }

    /* Hide all code-related elements */
    pre, code, .language-python, .language-html, .language-css, .language-javascript {
        display: none !important;
        visibility: hidden !important;
    }

    /* Hide elements containing code */
    .element-container:has(pre),
    .element-container:has(code),
    .element-container:has(.stCodeBlock),
    .element-container:has([data-testid="stCodeBlock"]) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Hide code blocks in specific containers */
    .hide-code-blocks .stCodeBlock,
    .hide-code-blocks div[data-testid="stCodeBlock"] {
        display: none !important;
    }

    /* Hide HTML tags and code that might be displayed in the UI */
    .element-container:has(p:contains("<div")),
    .element-container:has(p:contains("<button")),
    .element-container:has(p:contains("<span")),
    .element-container:has(p:contains("<a")),
    .element-container:has(p:contains("<p")),
    .element-container:has(p:contains("</")),
    .element-container:has(p:contains("style=")),
    .element-container:has(p:contains("class=")),
    .element-container:has(p:contains("onclick=")),
    .element-container:has(p:contains("href=")) {
        display: none !important;
    }

    /* Additional selectors to hide raw HTML code */
    pre:has(code:contains("<div")),
    pre:has(code:contains("<button")),
    pre:has(code:contains("<span")),
    pre:has(code:contains("<a")),
    pre:has(code:contains("<p")),
    pre:has(code:contains("</")),
    pre:has(code:contains("style=")),
    pre:has(code:contains("class=")),
    pre:has(code:contains("onclick=")),
    pre:has(code:contains("href=")) {
        display: none !important;
    }

    /* Hide any element containing HTML code */
    .hidden {
        display: none !important;
    }

    /* Target specific code blocks visible in the UI */
    div.element-container:has(p:contains("<div style=\"display: flex")),
    div.element-container:has(p:contains("<button onclick=")),
    div.element-container:has(p:contains("<div class=\"feature-icon\"")),
    div.element-container:has(p:contains("<div style=\"width: 36px")),
    div.element-container:has(p:contains("<div style=\"font-weight: 600")),
    div.element-container:has(p:contains("<div style=\"color: #6c757d")),
    div.element-container:has(p:contains("<p class=\"feature-description\"")),
    div.element-container:has(p:contains("<p style=\"text-align: center")),
    div.element-container:has(p:contains("<div style=\"margin-top: 1rem")),
    div.element-container:has(p:contains("<a href=\"/")),
    /* Target specific elements from the LLM Verification page */
    div.element-container:has(p:contains("<div class=\"verification-card\"")),
    div.element-container:has(p:contains("<div style=\"background-color: #1e293b")),
    div.element-container:has(p:contains("<div style=\"display: flex; justify-content: space-between")),
    div.element-container:has(p:contains("<div style=\"text-align: center; flex: 1")),
    div.element-container:has(p:contains("<div style=\"background-color: #5BC0BE; color: white")),
    div.element-container:has(p:contains("<div style=\"font-size: 0.8rem; color: #6c757d")),
    div.element-container:has(p:contains("<div style=\"font-size: 0.9rem; color: #6c757d")),
    div.element-container:has(p:contains("<span style=\"color: #5BC0BE")),
    div.element-container:has(p:contains("<div class=\"confidence-meter\"")),
    div.element-container:has(p:contains("<div class=\"fill\"")),
    div.element-container:has(p:contains("<div style=\"background-color: #0f172a")),
    div.element-container:has(p:contains("<h3 style=\"margin-top: 0; color: #e2e8f0")),
    div.element-container:has(p:contains("<p style=\"color: #e2e8f0")),
    /* Target specific elements from the home page */
    div.element-container:has(p:contains("<div class=\"hero-container\"")),
    div.element-container:has(p:contains("<div style=\"animation: fadeIn")),
    div.element-container:has(p:contains("<div style=\"font-size: 1.2rem; color: white")),
    div.element-container:has(p:contains("<h1 style=\"font-size: 3.5rem")),
    div.element-container:has(p:contains("<div class=\"animated-banner\"")),
    div.element-container:has(p:contains("<div class=\"banner-content\"")),
    div.element-container:has(p:contains("<span class=\"banner-icon\"")),
    div.element-container:has(p:contains("<span class=\"banner-text\"")),
    div.element-container:has(p:contains("<p style=\"font-size: 1.5rem; font-weight: 400")),
    div.element-container:has(p:contains("<style>")),
    div.element-container:has(p:contains("@keyframes")) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>

    <script>
    // Enhanced function to hide HTML code blocks and raw code
    function hideHtmlCodeBlocks() {
        // Function to check if an element is the title or banner we want to keep
        function isBanner(element) {
            // Check if it's our title or banner element
            if (element.classList && (
                element.classList.contains('main-title-container') ||
                element.classList.contains('main-title-text') ||
                element.classList.contains('main-header')
            )) {
                return true;
            }

            // Check by content
            if (element.innerHTML && (
                element.innerHTML.includes('background: linear-gradient(90deg, #5BC0BE, #3A506B, #5BC0BE') ||
                element.innerHTML.includes('📰 Fake News Detection System') ||
                element.innerHTML.includes('Powered by AI & Machine Learning')
            )) {
                return true;
            }

            return false;
        }

        // Find all elements that might contain code
        const elements = document.querySelectorAll('p, pre, code, div.stCodeBlock, div[data-testid="stCodeBlock"], div.element-container');

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

            // Skip the banner we want to keep visible
            if (isBanner(el)) return;

            // Check if the element is inside a banner container
            if (el.closest('.element-container') && isBanner(el.closest('.element-container'))) return;

            // Check if element contains code patterns
            const containsCode = codePatterns.some(pattern => text.includes(pattern));

            if (containsCode) {
                // Find the closest container to hide
                const container = el.closest('.element-container') || el;

                // Skip if this is our banner
                if (isBanner(container)) return;

                container.classList.add('hidden');
                container.style.display = 'none';

                // If it's a code block, hide it directly
                if (el.classList.contains('stCodeBlock') ||
                    el.hasAttribute('data-testid') && el.getAttribute('data-testid') === 'stCodeBlock') {
                    el.style.display = 'none';
                }
            }
        });

        // Specifically target code blocks
        const codeBlocks = document.querySelectorAll('.stCodeBlock, [data-testid="stCodeBlock"]');
        codeBlocks.forEach(block => {
            block.style.display = 'none';
        });

        // Target specific patterns seen in the screenshots
        hideSpecificCodeBlocks();

        // Make sure our title and banner are visible
        const titleContainers = document.querySelectorAll('.main-title-container, div[style*="background: linear-gradient(90deg, #5BC0BE, #3A506B, #5BC0BE"]');
        titleContainers.forEach(element => {
            // Make the element itself visible
            element.classList.remove('hidden');
            element.style.display = element.classList.contains('main-title-container') ? 'flex' : 'block';
            element.style.visibility = 'visible';
            element.style.height = 'auto';
            element.style.opacity = '1';
            element.style.margin = '0';
            element.style.padding = element.classList.contains('main-title-container') ? '0' : '10px 20px';

            // Make its container visible
            const container = element.closest('.element-container');
            if (container) {
                container.classList.remove('hidden');
                container.style.display = 'block';
                container.style.visibility = 'visible';
                container.style.height = 'auto';
                container.style.opacity = '1';
                container.style.margin = '0';
                container.style.padding = '0';
            }
        });

        // Also ensure the title text is visible
        const titleTexts = document.querySelectorAll('.main-title-text, .main-header, h1');
        titleTexts.forEach(text => {
            text.classList.remove('hidden');
            text.style.display = 'block';
            text.style.visibility = 'visible';
            text.style.opacity = '1';

            const container = text.closest('.element-container');
            if (container) {
                container.classList.remove('hidden');
                container.style.display = 'block';
                container.style.visibility = 'visible';
                container.style.height = 'auto';
                container.style.opacity = '1';
                container.style.margin = '0';
                container.style.padding = '0';
            }
        });
    }

    // Function to hide specific code blocks visible in the screenshots
    function hideSpecificCodeBlocks() {
        // Specific text patterns to look for
        const specificPatterns = [
            '<div style="display: flex',
            '<button onclick=',
            'parent.window.location.href=',
            'style="background-color:',
            'Try Prediction',
            'LLM Verification',
            '<div class="feature-icon"',
            '<p class="feature-description"',
            '<div style="width: 36px',
            '<div style="font-weight: 600',
            '<div style="color: #6c757d',
            '<p style="text-align: center',
            '<div style="margin-top: 1rem',
            '<a href="/',
            // LLM Verification page patterns
            '<div class="verification-card"',
            '<div style="background-color: #1e293b',
            '<div style="display: flex; justify-content: space-between',
            '<div style="text-align: center; flex: 1',
            '<div style="background-color: #5BC0BE; color: white',
            '<div style="font-size: 0.8rem; color: #6c757d',
            '<div style="font-size: 0.9rem; color: #6c757d',
            '<span style="color: #5BC0BE',
            '<div class="confidence-meter"',
            '<div class="fill"',
            '<div style="background-color: #0f172a',
            '<h3 style="margin-top: 0; color: #e2e8f0',
            '<p style="color: #e2e8f0',
            // Home page patterns
            '<div class="hero-container"',
            '<div style="animation: fadeIn',
            '<div style="font-size: 1.2rem; color: white',
            '<h1 style="font-size: 3.5rem',
            '<div class="animated-banner"',
            '<div class="banner-content"',
            '<span class="banner-icon"',
            '<span class="banner-text"',
            '<p style="font-size: 1.5rem; font-weight: 400',
            '<style>',
            '@keyframes'
        ];

        // Using the isBanner function from the parent scope

        // Find all paragraphs and divs
        const elements = document.querySelectorAll('p, div');

        elements.forEach(el => {
            const text = el.innerText || el.textContent;
            if (!text) return;

            // Skip the banner we want to keep visible
            if (isBanner(el)) return;

            // Check if the element is inside a banner container
            if (el.closest('.element-container') && isBanner(el.closest('.element-container'))) return;

            // Check if element contains any specific patterns
            const containsPattern = specificPatterns.some(pattern => text.includes(pattern));

            if (containsPattern) {
                // Find the closest container to hide
                const container = el.closest('.element-container') || el;

                // Skip if this is our banner
                if (isBanner(container)) return;

                container.classList.add('hidden');
                container.style.display = 'none';
            }
        });
    }

    // Function to rename "app" to "Home" and hide specific sidebar items
    function renameAppToHome() {
        // Find the sidebar navigation
        const sidebarNav = document.querySelector('[data-testid="stSidebarNav"]');
        if (sidebarNav) {
            // Find the first link which is typically the "app" link
            const firstLink = sidebarNav.querySelector('li:first-child a p');
            if (firstLink && firstLink.textContent.trim() === 'app') {
                // Change the text to "Home"
                firstLink.textContent = 'Home';
            }

            // Also try to find links by href
            const homeLinks = sidebarNav.querySelectorAll('a[href="/"], a[href="/app"]');
            homeLinks.forEach(link => {
                // Find any span or p elements inside this link
                const textElements = link.querySelectorAll('span, p');
                textElements.forEach(element => {
                    if (element.textContent.trim() === 'app') {
                        element.textContent = 'Home';
                    }
                });
            });

            // Hide Model Explanation and Prediction from sidebar - aggressive approach

            // Method 1: Hide by position (3rd and 4th items)
            const navItems = sidebarNav.querySelectorAll('li');
            if (navItems.length >= 4) {
                // Hide the 3rd and 4th items (Model Explanation and Prediction)
                if (navItems[2]) navItems[2].style.display = 'none';
                if (navItems[3]) navItems[3].style.display = 'none';
            }

            // Method 2: Hide by text content
            const allLinks = sidebarNav.querySelectorAll('li a');
            allLinks.forEach(link => {
                // Check if the link text or href contains Model Explanation or Prediction
                const linkText = link.textContent.trim();
                const href = link.getAttribute('href') || '';

                if (linkText.includes('Model Explanation') ||
                    linkText.includes('Prediction') ||
                    href.includes('Model_Explanation') ||
                    href.includes('Prediction')) {
                    // Find the parent li element and hide it
                    const parentLi = link.closest('li');
                    if (parentLi) {
                        parentLi.style.display = 'none';
                        parentLi.style.visibility = 'hidden';
                        parentLi.style.height = '0';
                        parentLi.style.overflow = 'hidden';
                        parentLi.style.margin = '0';
                        parentLi.style.padding = '0';
                    }

                    // Also hide the link itself
                    link.style.display = 'none';
                    link.style.visibility = 'hidden';
                }
            });

            // Method 3: Direct DOM removal
            const itemsToRemove = [];
            navItems.forEach(item => {
                const itemText = item.textContent.trim();
                if (itemText.includes('Model Explanation') || itemText.includes('Prediction')) {
                    itemsToRemove.push(item);
                }
            });

            // Remove the items from the DOM
            itemsToRemove.forEach(item => {
                try {
                    item.parentNode.removeChild(item);
                } catch (e) {
                    // Fallback to hiding if removal fails
                    item.style.display = 'none';
                }
            });
        }
    }

    // Run when the page loads and after any Streamlit rerun
    document.addEventListener('DOMContentLoaded', function() {
        // Initial run
        hideHtmlCodeBlocks();
        renameAppToHome();

        // Run again after a short delay to catch elements that might load later
        setTimeout(hideHtmlCodeBlocks, 500);
        setTimeout(hideHtmlCodeBlocks, 1000);

        // Run the rename function a few times with reasonable delays
        setTimeout(renameAppToHome, 100);
        setTimeout(renameAppToHome, 500);
        setTimeout(renameAppToHome, 1000);

        // Set up a simple interval to check occasionally
        const renameInterval = setInterval(renameAppToHome, 2000);

        // Clear the interval after 10 seconds
        setTimeout(() => {
            clearInterval(renameInterval);
        }, 10000);
    });

    // Use MutationObserver to detect DOM changes
    const observer = new MutationObserver(function(mutations) {
        // Run our functions when DOM changes
        hideHtmlCodeBlocks();
        renameAppToHome();

        // Check if any mutations affected the sidebar navigation
        const sidebarMutation = mutations.some(mutation => {
            return mutation.target.closest && mutation.target.closest('[data-testid="stSidebarNav"]');
        });

        // If sidebar was affected, make sure to run our function again after a short delay
        if (sidebarMutation) {
            setTimeout(renameAppToHome, 100);
        }
    });

    // Start observing with a more comprehensive configuration
    observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true
    });
    </script>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Get root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# No duplicate functions needed as they are imported from utils.py

# Main content
if st.session_state.current_page == 'Home':
    # Add title and banner at the top of the page
    st.markdown("""
    <style>
    /* Force the title to be at the top of the page */
    div.block-container {
        padding-top: 0 !important;
    }

    /* Remove margins from the title container */
    div.element-container:first-of-type {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Ensure the title is at the top */
    header[data-testid="stHeader"] {
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Fix the main container padding */
    .main .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Make sure the title is visible */
    .main-title-container {
        display: block !important;
        visibility: visible !important;
        height: auto !important;
        opacity: 1 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Ensure the title text is visible */
    .main-title-text {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    </style>

    <div class="main-title-container" style="display: flex; flex-direction: column; gap: 0; margin: 0; padding: 0;">
        <h1 class="main-title-text main-header" style="margin: 0; padding: 0; font-size: 3rem; font-weight: 700; text-align: center; background: linear-gradient(90deg, #3A506B, #5BC0BE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            📰 Fake News Detection System
        </h1>

        <div style="background: linear-gradient(90deg, #5BC0BE, #3A506B, #5BC0BE);
                    background-size: 200% 200%;
                    animation: gradientShift 3s ease infinite;
                    padding: 10px 20px;
                    border-radius: 50px;
                    text-align: center;
                    margin: 0;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                <div style="font-size: 20px; animation: pulse 2s infinite ease-in-out;">🔍</div>
                <div style="font-weight: 600; letter-spacing: 1px; color: white; font-size: 18px;">
                    Powered by AI & Machine Learning
                </div>
                <div style="font-size: 20px; animation: pulse 2s infinite ease-in-out;">🤖</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar with improved styling
    with st.sidebar:
        # Logo and title
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://img.icons8.com/fluency/96/000000/news.png", width=60)
        with col2:
            st.markdown("<h2 style='margin-top:0.5rem;'>Fake News<br>Detection</h2>", unsafe_allow_html=True)

        # Add direct style tag to hide Model Explanation and Prediction
        st.markdown("""
        <style>
        /* Direct style to hide Model Explanation and Prediction in sidebar */
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:nth-child(3),
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:nth-child(4),
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="Model_Explanation"],
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[href*="Prediction"],
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:has(a[href*="Model_Explanation"]),
        section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li:has(a[href*="Prediction"]) {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            width: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 📊 Navigation")

        # System Status with custom indicators
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🖥️ System Status")

        status = check_system_status(ROOT_DIR)

        # Models status with custom indicator
        if status["models"]:
            render_status_indicator(True, "Models loaded successfully", "success")
        else:
            render_status_indicator(False, "Models not found", "error")

        # Data status with custom indicator
        if status["data"]:
            render_status_indicator(True, "Dataset ready", "success")
        else:
            render_status_indicator(False, "Dataset not found", "error")

        # Check Ollama status
        ollama_status = check_ollama_status()
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 🤖 LLM Verification")

        # Ollama status with custom indicator
        if ollama_status["running"]:
            render_status_indicator(True, "Ollama running", "success")

            if ollama_status["models"]:
                model_list = ", ".join(ollama_status["models"])
                st.info(f"Available models: {model_list}")
            else:
                render_status_indicator(False, "No models found. Please pull a model.", "warning")
        else:
            render_status_indicator(False, "Ollama not running", "error")

            if "error" in ollama_status:
                st.error(ollama_status['error'])

    # Load and display dataset statistics with improved styling
    try:
        with st.spinner("Loading dataset statistics..."):
            df = load_data(ROOT_DIR)
            if df is not None:
                # Add a brief description
                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(28, 37, 65, 0.05), rgba(91, 192, 190, 0.05));
                            padding: 1.5rem;
                            border-radius: 10px;
                            margin-top: 1.5rem;
                            text-align: center;">
                    <p style="font-size: 1.2rem; color: #3A506B; max-width: 800px; margin: 0 auto;">
                        An advanced AI-powered platform to identify and analyze potentially misleading news articles
                        using machine learning and natural language processing
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # Add buttons for navigation - hiding Model Explanation and Prediction
                buttons = [
                    # 'Try Prediction' button is hidden
                    # {
                    #     'label': 'Try Prediction',
                    #     'url': '/Prediction',
                    #     'icon': '🎯'
                    # },
                    {
                        'label': 'LLM Verification',
                        'url': '/LLM_Verification',
                        'icon': '🤖'
                    }
                ]

                render_button_group(buttons)

                # Metrics in cards with hover effects
                st.markdown("<div style='margin: 2rem 0;'></div>", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 3rem; font-weight: 700; color: #3A506B; margin-bottom: 0.5rem;">{len(df):,}</div>
                        <div style="font-size: 1.2rem; color: #6c757d; font-weight: 500;">Total Articles</div>
                        <div style="width: 50px; height: 4px; background: #5BC0BE; margin-top: 1rem;"></div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    real_percent = (df['label'] == 'Real').mean() * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 3rem; font-weight: 700; color: #2E7D32; margin-bottom: 0.5rem;">{real_percent:.1f}%</div>
                        <div style="font-size: 1.2rem; color: #6c757d; font-weight: 500;">Real News</div>
                        <div style="width: 50px; height: 4px; background: #2E7D32; margin-top: 1rem;"></div>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    fake_percent = (df['label'] == 'Fake').mean() * 100
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 3rem; font-weight: 700; color: #C62828; margin-bottom: 0.5rem;">{fake_percent:.1f}%</div>
                        <div style="font-size: 1.2rem; color: #6c757d; font-weight: 500;">Fake News</div>
                        <div style="width: 50px; height: 4px; background: #C62828; margin-top: 1rem;"></div>
                    </div>
                    """, unsafe_allow_html=True)

                # Distribution plot with improved styling
                st.markdown("<div style='margin: 3rem 0 1rem 0;'></div>", unsafe_allow_html=True)

                col1, col2 = st.columns([2, 3])

                with col1:
                    st.markdown("""
                    <h2 style="margin-top: 0;">📊 News Distribution</h2>
                    <p style="color: #6c757d; margin-bottom: 2rem;">
                        Breakdown of real vs. fake news articles in our dataset. This balanced distribution ensures our models are trained without bias.
                    </p>
                    """, unsafe_allow_html=True)

                with col2:
                    fig = px.pie(
                        df,
                        names='label',
                        title=None,
                        color_discrete_sequence=['#5BC0BE', '#FF6B6B'],
                        hole=0.4
                    )

                    fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#FFFFFF', width=2))
                    )

                    fig.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0),
                        height=300,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                # Divider
                st.markdown("<hr>", unsafe_allow_html=True)

                # Project overview with enhanced styling
                st.markdown("""
                <div class="feature-highlight">
                    <h2 style="text-align: center; color: #3A506B; margin-bottom: 2rem;">
                        <span style="color: #5BC0BE;">🎯</span> Key Features
                    </h2>

                    <p style="text-align: center; font-size: 1.1rem; color: #3A506B; margin-bottom: 2.5rem; max-width: 800px; margin-left: auto; margin-right: auto;">
                        This Fake News Detection System uses advanced machine learning and explainable AI techniques
                        to help identify potentially misleading news articles through multiple analysis methods:
                    </p>

                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; margin-bottom: 1.5rem;">
                        <div class="feature-card">
                            <div class="feature-icon">📝</div>
                            <div class="feature-title">Stylometric Analysis</div>
                            <p class="feature-description">
                                Analyzes text patterns and writing style to identify characteristics common in fake news, such as excessive punctuation, capitalization, and sentence structure.
                            </p>
                            <div style="margin-top: 1rem;">
                                <a href="/Text_Analysis" style="color: #5BC0BE; text-decoration: none; font-weight: 500; font-size: 0.9rem; display: flex; align-items: center; gap: 0.25rem;">
                                    Explore Analysis <span style="font-size: 1.2rem;">→</span>
                                </a>
                            </div>
                        </div>

                        <div class="feature-card">
                            <div class="feature-icon">😊</div>
                            <div class="feature-title">Sentiment Analysis</div>
                            <p class="feature-description">
                                Examines emotional tone and subjectivity in text to detect bias and emotional manipulation often present in fake news articles.
                            </p>
                            <div style="margin-top: 1rem;">
                                <a href="/Text_Analysis" style="color: #5BC0BE; text-decoration: none; font-weight: 500; font-size: 0.9rem; display: flex; align-items: center; gap: 0.25rem;">
                                    View Sentiment <span style="font-size: 1.2rem;">→</span>
                                </a>
                            </div>
                        </div>

                        <div class="feature-card">
                            <div class="feature-icon">🤖</div>
                            <div class="feature-title">Machine Learning</div>
                            <p class="feature-description">
                                Uses multiple models for robust prediction, combining the strengths of different algorithms to improve accuracy and reduce false positives.
                            </p>
                            <div style="margin-top: 1rem; display: none;">
                                <a href="/Prediction" style="color: #5BC0BE; text-decoration: none; font-weight: 500; font-size: 0.9rem; display: flex; align-items: center; gap: 0.25rem;">
                                    Try Prediction <span style="font-size: 1.2rem;">→</span>
                                </a>
                            </div>
                        </div>

                        <div class="feature-card">
                            <div class="feature-icon">🔍</div>
                            <div class="feature-title">Explainable AI</div>
                            <p class="feature-description">
                                Provides transparent explanations for why an article might be classified as fake, highlighting the specific words and patterns that influenced the decision.
                            </p>
                            <div style="margin-top: 1rem; display: none;">
                                <a href="/Model_Explanation" style="color: #5BC0BE; text-decoration: none; font-weight: 500; font-size: 0.9rem; display: flex; align-items: center; gap: 0.25rem;">
                                    See Explanations <span style="font-size: 1.2rem;">→</span>
                                </a>
                            </div>
                        </div>
                    </div>

                    <div class="feature-card" style="margin-top: 1.5rem;">
                        <div style="display: flex; align-items: flex-start; gap: 1.5rem;">
                            <div class="feature-icon">🌐</div>
                            <div style="flex: 1;">
                                <div class="feature-title">LLM Verification</div>
                                <p class="feature-description">
                                    Uses a local LLM with internet search capabilities to verify news claims against reliable sources, providing an additional layer of fact-checking.
                                </p>
                                <div style="margin-top: 1rem;">
                                    <a href="/LLM_Verification" style="color: #5BC0BE; text-decoration: none; font-weight: 500; font-size: 0.9rem; display: flex; align-items: center; gap: 0.25rem;">
                                        Verify with LLM <span style="font-size: 1.2rem;">→</span>
                                    </a>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Quick Start Guide with enhanced styling
                st.markdown("<div style='margin: 3rem 0 1rem 0;'></div>", unsafe_allow_html=True)

                st.markdown("""
                <div style="background: linear-gradient(135deg, rgba(28, 37, 65, 0.05), rgba(91, 192, 190, 0.05)); padding: 2.5rem; border-radius: 10px; position: relative; overflow: hidden;">
                    <div style="position: absolute; top: 0; right: 0; width: 150px; height: 150px; background: linear-gradient(135deg, rgba(91, 192, 190, 0.2), rgba(28, 37, 65, 0.2)); border-radius: 0 0 0 150px; z-index: 0;"></div>

                    <div style="position: relative; z-index: 1;">
                        <h2 style="text-align: center; color: #3A506B; margin-bottom: 2rem; display: flex; align-items: center; justify-content: center; gap: 0.5rem;">
                            <span style="color: #5BC0BE; font-size: 2rem;">🚀</span>
                            <span>Quick Start Guide</span>
                        </h2>

                        <p style="text-align: center; color: #3A506B; max-width: 700px; margin: 0 auto 2.5rem auto;">
                            Follow these simple steps to get started with the Fake News Detection System and make the most of its features:
                        </p>

                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;">
                            <div class="feature-card" style="border-left: 4px solid #5BC0BE; padding-left: 1.5rem;">
                                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                    <div style="background-color: #5BC0BE; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 1.2rem;">1</div>
                                    <div class="feature-title" style="margin: 0;">Text Analysis</div>
                                </div>
                                <p class="feature-description">
                                    Explore patterns in the dataset to understand the characteristics that differentiate real from fake news.
                                </p>
                                <a href="/Text_Analysis" class="btn-primary" style="display: inline-block; background-color: #5BC0BE; color: white; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; font-weight: 500; margin-top: 1rem; transition: all 0.2s ease;">
                                    Start Analysis
                                </a>
                            </div>

                            <!-- Model Explanation card is hidden -->
                            <div class="feature-card" style="border-left: 4px solid #5BC0BE; padding-left: 1.5rem; display: none;">
                                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                    <div style="background-color: #5BC0BE; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 1.2rem;">2</div>
                                    <div class="feature-title" style="margin: 0;">Model Explanation</div>
                                </div>
                                <p class="feature-description">
                                    Understand how our AI works and what factors influence its decisions through interactive visualizations.
                                </p>
                                <a href="/Model_Explanation" class="btn-primary" style="display: inline-block; background-color: #5BC0BE; color: white; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; font-weight: 500; margin-top: 1rem; transition: all 0.2s ease;">
                                    Explore Models
                                </a>
                            </div>

                            <!-- Prediction card is hidden -->
                            <div class="feature-card" style="border-left: 4px solid #5BC0BE; padding-left: 1.5rem; display: none;">
                                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                    <div style="background-color: #5BC0BE; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 1.2rem;">3</div>
                                    <div class="feature-title" style="margin: 0;">Prediction</div>
                                </div>
                                <p class="feature-description">
                                    Analyze news articles to determine if they're likely to be fake or real using our machine learning models.
                                </p>
                                <a href="/Prediction" class="btn-primary" style="display: inline-block; background-color: #5BC0BE; color: white; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; font-weight: 500; margin-top: 1rem; transition: all 0.2s ease;">
                                    Make Predictions
                                </a>
                            </div>

                            <div class="feature-card" style="border-left: 4px solid #5BC0BE; padding-left: 1.5rem;">
                                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                                    <div style="background-color: #5BC0BE; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 1.2rem;">4</div>
                                    <div class="feature-title" style="margin: 0;">LLM Verification</div>
                                </div>
                                <p class="feature-description">
                                    Verify news with internet search to fact-check claims against reliable sources using local LLM technology.
                                </p>
                                <a href="/LLM_Verification" class="btn-primary" style="display: inline-block; background-color: #5BC0BE; color: white; padding: 0.5rem 1rem; border-radius: 4px; text-decoration: none; font-weight: 500; margin-top: 1rem; transition: all 0.2s ease;">
                                    Verify with LLM
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("Please ensure the dataset file is present in the data directory.")

    # Modern footer with contact information and social links
    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("""
        <div style="padding: 1.5rem 0;">
            <h3 style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                <span style="color: #5BC0BE;">📬</span> Need Help?
            </h3>
            <p style="color: #6c757d; margin-bottom: 1rem;">
                Our team is here to help you with any questions or issues you might have.
                Feel free to reach out through any of the channels below.
            </p>

            <div style="display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 36px; height: 36px; border-radius: 50%; background-color: #5BC0BE; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.2rem;">
                        📧
                    </div>
                    <div>
                        <div style="font-weight: 600; color: #3A506B;">Email</div>
                        <div style="color: #6c757d; font-size: 0.9rem;">support@fakenewsdetection.com</div>
                    </div>
                </div>

                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 36px; height: 36px; border-radius: 50%; background-color: #5BC0BE; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.2rem;">
                        📱
                    </div>
                    <div>
                        <div style="font-weight: 600; color: #3A506B;">Phone</div>
                        <div style="color: #6c757d; font-size: 0.9rem;">+1 (555) 123-4567</div>
                    </div>
                </div>

                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 36px; height: 36px; border-radius: 50%; background-color: #5BC0BE; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.2rem;">
                        🌐
                    </div>
                    <div>
                        <div style="font-weight: 600; color: #3A506B;">Website</div>
                        <div style="color: #6c757d; font-size: 0.9rem;">www.fakenewsdetection.com</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="padding: 1.5rem 0;">
            <h4 style="color: #1C2541; margin-bottom: 1rem; font-weight: 700; font-size: 1.2rem;">Quick Links</h4>
        </div>
        """, unsafe_allow_html=True)

        # Create clickable buttons for quick links using switch_page
        if st.button("📊 Text Analysis", key="text_analysis_btn", help="Go to Text Analysis page"):
            st.switch_page("pages/1_Text_Analysis.py")

        # Model Explanation button is hidden but functionality is preserved
        if False:  # This condition ensures the button is not displayed
            if st.button("🔍 Model Explanation", key="model_explanation_btn", help="Go to Model Explanation page"):
                st.switch_page("pages/2_Model_Explanation.py")

        # Prediction button is hidden but functionality is preserved
        if False:  # This condition ensures the button is not displayed
            if st.button("🔮 Prediction", key="prediction_btn", help="Go to Prediction page"):
                st.switch_page("pages/3_Prediction.py")

        if st.button("🌐 LLM Verification", key="llm_verification_btn", help="Go to LLM Verification page"):
            st.switch_page("pages/4_LLM_Verification.py")

    with col3:
        st.markdown("""
        <div style="padding: 1.5rem 0;">
            <h4 style="color: #1C2541; margin-bottom: 1rem; font-weight: 700; font-size: 1.2rem;">Resources</h4>
        </div>
        """, unsafe_allow_html=True)

        # Check if resource pages exist
        try:
            # Create clickable buttons for resources using switch_page
            if st.button("📚 Documentation", key="documentation_btn", help="View Documentation"):
                st.switch_page("pages/5_Documentation.py")

            if st.button("🎓 Tutorials", key="tutorials_btn", help="View Tutorials"):
                st.switch_page("pages/6_Tutorials.py")

            if st.button("❓ FAQ", key="faq_btn", help="View Frequently Asked Questions"):
                st.switch_page("pages/7_FAQ.py")

            if st.button("🔒 Privacy Policy", key="privacy_policy_btn", help="View Privacy Policy"):
                st.switch_page("pages/8_Privacy_Policy.py")
        except Exception as e:
            st.warning("Some resource pages may not be available yet. We're working on adding them soon!")
            # Log the error for debugging
            print(f"Error accessing resource pages: {str(e)}")

    # Copyright footer
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-top: 1px solid #e0e0e0; margin-top: 1rem;">
        <p style="color: #6c757d; font-size: 0.9rem;">
            © 2023 Fake News Detection System. All rights reserved.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main function is defined below

def main():
    st.title("📰 Fake News Detection System")

    # Add a banner under the title
    st.markdown("""
    <div style="background: linear-gradient(90deg, #5BC0BE, #3A506B, #5BC0BE);
                background-size: 200% 200%;
                animation: gradientShift 3s ease infinite;
                padding: 10px 20px;
                border-radius: 50px;
                text-align: center;
                margin: 10px 0 20px 0;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
        <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
            <div style="font-size: 20px; animation: pulse 2s infinite ease-in-out;">🔍</div>
            <div style="font-weight: 600; letter-spacing: 1px; color: white; font-size: 18px;">
                Powered by AI & Machine Learning
            </div>
            <div style="font-size: 20px; animation: pulse 2s infinite ease-in-out;">🤖</div>
        </div>
    </div>

    <style>
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data(ROOT_DIR)
    if df is None:
        st.stop()

    # Basic stats
    st.write("Dataset loaded successfully!")
    st.write(f"Total articles: {len(df):,}")

    # Simple visualization
    st.subheader("Distribution of Articles")
    fig = px.pie(df, names='label', title='Real vs Fake News Distribution')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()