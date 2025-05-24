"""
HTML Renderer Component for Streamlit

This module provides functions to render HTML content in Streamlit without showing the raw HTML code.
It uses Streamlit's components and HTML/CSS/JS to properly render HTML content.
"""

import streamlit as st
import base64
import re

def render_html(html_content, height=None, scrolling=False):
    """
    Renders HTML content in an iframe to prevent raw HTML code from being displayed.

    Parameters:
    -----------
    html_content : str
        The HTML content to render
    height : int or None
        The height of the iframe in pixels. If None, it will adjust to content.
    scrolling : bool
        Whether to enable scrolling in the iframe

    Returns:
    --------
    None
    """
    # Encode the HTML content
    encoded_content = base64.b64encode(html_content.encode()).decode()

    # Set default height if not provided
    if height is None:
        # Estimate height based on content length (rough heuristic)
        height = min(max(100, len(html_content) // 5), 800)

    # Set scrolling attribute
    scroll_attr = "yes" if scrolling else "no"

    # Create an iframe with the encoded content
    iframe_html = f"""
    <iframe srcdoc="{encoded_content}"
            width="100%"
            height="{height}px"
            frameborder="0"
            scrolling="{scroll_attr}"
            style="border: none; width: 100%;">
    </iframe>
    """

    # Use st.components.v1.html to render the iframe
    st.components.v1.html(iframe_html, height=height+20)

def render_styled_html(html_content, height=None, scrolling=False):
    """
    Renders HTML content with default styling in an iframe.

    Parameters:
    -----------
    html_content : str
        The HTML content to render
    height : int or None
        The height of the iframe in pixels. If None, it will adjust to content.
    scrolling : bool
        Whether to enable scrolling in the iframe

    Returns:
    --------
    None
    """
    # Add default styling to the HTML content
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                color: white;
                background-color: transparent;
                margin: 0;
                padding: 0;
            }}

            * {{
                box-sizing: border-box;
            }}

            h1, h2, h3, h4, h5, h6 {{
                color: white;
                margin-top: 0.5em;
                margin-bottom: 0.5em;
            }}

            p {{
                margin-bottom: 1em;
                line-height: 1.5;
            }}

            a {{
                color: #5BC0BE;
                text-decoration: none;
            }}

            a:hover {{
                text-decoration: underline;
            }}

            .container {{
                width: 100%;
                padding: 0;
            }}

            .btn {{
                display: inline-block;
                padding: 0.5rem 1rem;
                background-color: #5BC0BE;
                color: white;
                border-radius: 4px;
                text-decoration: none;
                font-weight: 500;
                cursor: pointer;
                border: none;
                transition: all 0.2s ease;
            }}

            .btn:hover {{
                background-color: #3A506B;
                transform: translateY(-2px);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            {html_content}
        </div>
    </body>
    </html>
    """

    # Render the styled HTML
    render_html(styled_html, height=height, scrolling=scrolling)

def render_button_group(buttons, height=80):
    """
    Renders a group of buttons with proper styling.

    Parameters:
    -----------
    buttons : list of dict
        List of button dictionaries with 'label', 'url', and optional 'icon' keys
    height : int
        The height of the button container

    Returns:
    --------
    None
    """
    # Create direct HTML for buttons without using iframe/base64
    buttons_html = '<div style="display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center;">'

    for button in buttons:
        icon = button.get('icon', '')
        label = button.get('label', 'Button')
        url = button.get('url', '#')

        buttons_html += f"""
        <a href="{url}" style="display: inline-block; padding: 0.5rem 1rem; background-color: #5BC0BE; color: white;
                               border-radius: 4px; text-decoration: none; font-weight: 500; cursor: pointer;
                               border: none; transition: all 0.2s ease;">
            {icon} {label}
        </a>
        """

    buttons_html += '</div>'

    # Use direct Streamlit markdown instead of the iframe approach
    st.markdown(buttons_html, unsafe_allow_html=True)

def render_status_indicator(status, message, type="success", height=60):
    """
    Renders a status indicator with an icon and message.

    Parameters:
    -----------
    status : bool
        The status to display (True for success, False for error)
    message : str
        The message to display
    type : str
        The type of indicator ('success', 'error', 'warning')
    height : int
        The height of the indicator

    Returns:
    --------
    None
    """
    # Define colors based on type
    colors = {
        "success": "#2E7D32",
        "error": "#C62828",
        "warning": "#FFD166"
    }

    # Define icons based on type
    icons = {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️"
    }

    # Get color and icon
    color = colors.get(type, colors["success"])
    icon = icons.get(type, icons["success"])

    # Create HTML for status indicator
    indicator_html = f"""
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {color};
                    box-shadow: 0 0 8px {color}80;"></div>
        <div style="font-weight: 500;">{message}</div>
    </div>
    """

    # Render the indicator
    render_styled_html(indicator_html, height=height)
