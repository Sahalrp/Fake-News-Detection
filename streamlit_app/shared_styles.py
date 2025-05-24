"""
Shared CSS styles for the Fake News Detection System.
This module provides consistent styling across all pages.
"""

def load_css():
    """
    Returns the shared CSS styles for the application.
    """
    return """
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

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1C2541 0%, #3A506B 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    section[data-testid="stSidebar"] hr {
        margin: 1.5rem 0;
        border-color: rgba(255, 255, 255, 0.2);
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
    .main-header {
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: #1C2541 !important;
        margin-bottom: 2rem !important;
        text-align: center;
        /* Remove gradient text that causes visibility issues */
        /*background: linear-gradient(90deg, #3A506B, #5BC0BE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;*/
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

    /* Feature highlight section */
    .feature-highlight {
        background: linear-gradient(135deg, rgba(91, 192, 190, 0.1), rgba(58, 80, 107, 0.1));
        border-radius: 10px;
        padding: 2rem;
        margin: 3rem 0;
    }

    .feature-card {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
    }

    .feature-icon {
        font-size: 2.5rem;
        color: #5BC0BE;
        margin-bottom: 1rem;
    }

    .feature-title {
        font-weight: 600;
        color: #3A506B;
        margin-bottom: 0.5rem;
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

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #3A506B !important;
        background-color: rgba(91, 192, 190, 0.1) !important;
        border-radius: 4px !important;
    }

    .streamlit-expanderContent {
        border-left: 1px solid #5BC0BE !important;
        padding-left: 1rem !important;
        margin-left: 0.5rem !important;
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

    /* Animation classes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 1s ease-in-out;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .slide-in {
        animation: slideIn 0.8s ease-out;
    }

    /* Page title styling */
    .page-title {
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-in-out;
    }

    .page-title h1 {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
        border-bottom: none !important;
        padding-bottom: 0 !important;
    }

    .page-title p {
        font-size: 1.2rem !important;
        color: #6c757d !important;
        max-width: 800px !important;
        margin: 0 auto !important;
    }

    /* Feature description */
    .feature-description {
        color: #6c757d;
        font-size: 0.9rem;
    }

    /* Verification card styling */
    .verification-card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    .verification-card.real {
        border-left: 5px solid #2E7D32;
    }

    .verification-card.fake {
        border-left: 5px solid #C62828;
    }

    .verification-card.unknown {
        border-left: 5px solid #FFD166;
    }

    .confidence-meter {
        height: 8px;
        background-color: #e0e0e0;
        border-radius: 4px;
        margin: 1rem 0;
        overflow: hidden;
    }

    .confidence-meter .fill {
        height: 100%;
        border-radius: 4px;
    }

    /* Claim box styling */
    .claim-box {
        background-color: rgba(91, 192, 190, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Search result styling */
    .search-result {
        margin-bottom: 1.5rem;
    }

    .search-result pre {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        overflow-x: auto;
    }

    /* Hide HTML tags and code that might be displayed in the UI - comprehensive approach */
    /* Hide all code blocks by default */
    .stCodeBlock,
    div[data-testid="stCodeBlock"],
    pre,
    code,
    .language-python,
    .language-html,
    .language-css,
    .language-javascript,
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
        opacity: 0 !important;
    }

    /* Hide elements containing HTML tags */
    p:contains("<div"),
    p:contains("<button"),
    p:contains("<span"),
    p:contains("<a"),
    p:contains("<p"),
    p:contains("</"),
    p:contains("<h"),
    p:contains("style="),
    p:contains("class="),
    p:contains("onclick="),
    p:contains("href="),
    div:contains("<div"),
    div:contains("<button"),
    div:contains("<span"),
    div:contains("<a"),
    div:contains("<p"),
    div:contains("</"),
    div:contains("<h"),
    div:contains("style="),
    div:contains("class="),
    div:contains("onclick="),
    div:contains("href=") {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
    }

    /* Hide parent containers of HTML code */
    .element-container:has(p:contains("<div")),
    .element-container:has(p:contains("<button")),
    .element-container:has(p:contains("<span")),
    .element-container:has(p:contains("<a")),
    .element-container:has(p:contains("<p")),
    .element-container:has(p:contains("</")),
    .element-container:has(p:contains("<h")),
    .element-container:has(p:contains("style=")),
    .element-container:has(p:contains("class=")),
    .element-container:has(p:contains("onclick=")),
    .element-container:has(p:contains("href=")),
    .element-container:has(div:contains("<div")),
    .element-container:has(div:contains("<button")),
    .element-container:has(div:contains("<span")),
    .element-container:has(div:contains("<a")),
    .element-container:has(div:contains("<p")),
    .element-container:has(div:contains("</")),
    .element-container:has(div:contains("<h")),
    .element-container:has(div:contains("style=")),
    .element-container:has(div:contains("class=")),
    .element-container:has(div:contains("onclick=")),
    .element-container:has(div:contains("href=")) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
    }

    /* Additional selectors to hide raw HTML code */
    pre:has(code:contains("<")),
    pre:has(code:contains(">")),
    pre:has(code:contains("style=")),
    pre:has(code:contains("class=")),
    pre:has(code:contains("function")),
    pre:has(code:contains("def ")),
    pre:has(code:contains("import ")),
    pre:has(code:contains("onclick=")),
    pre:has(code:contains("href=")) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
    }

    /* Hide code blocks in specific containers */
    .hide-code-blocks .stCodeBlock,
    .hide-code-blocks div[data-testid="stCodeBlock"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
    }

    /* Target specific elements that might be showing in the UI */
    .element-container:has(p:contains("onclick=")),
    .element-container:has(p:contains("<div style=")),
    .element-container:has(p:contains("<button onclick=")),
    .element-container:has(p:contains("</button>")),
    .element-container:has(p:contains("</div>")),
    .element-container:has(p:contains("<h3 style=")),
    .element-container:has(p:contains("<span style=")),
    .element-container:has(p:contains("<ol style=")),
    .element-container:has(p:contains("<ul style=")),
    .element-container:has(p:contains("<li style=")),
    .element-container:has(p:contains("@keyframes")),
    .element-container:has(p:contains("<style>")) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
    }

    /* Hide specific HTML patterns seen in the screenshots */
    .element-container:has(p:contains("display: grid; grid-template-columns:")),
    .element-container:has(p:contains("border-radius: 8px; padding:")),
    .element-container:has(p:contains("margin-top: 0; color: #3A506B;")),
    .element-container:has(p:contains("color: #6c757d; margin-bottom:")),
    .element-container:has(p:contains("background: linear-gradient")),
    .element-container:has(p:contains("display: flex; align-items:")),
    .element-container:has(p:contains("font-weight: 600; color:")),
    .element-container:has(p:contains("font-size: 0.9rem;")),
    .element-container:has(p:contains("padding: 1rem; background-color:")),
    .element-container:has(p:contains("margin-bottom: 1rem;")),
    .element-container:has(p:contains("gap: 1rem; flex-wrap:")),
    .element-container:has(p:contains("background-color: rgba(")) {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
    }
    </style>
    """
