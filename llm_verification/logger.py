"""
Logging module for the LLM verification system.
"""
import logging
import os
import sys
import json
import traceback
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Configure logging
log_file = os.path.join(logs_dir, f"llm_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Create a custom logger
logger = logging.getLogger("llm_verification")
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatters and add to handlers
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
console_format = logging.Formatter('%(levelname)s - %(message)s')

file_handler.setFormatter(file_format)
console_handler.setFormatter(console_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_request(url, method, headers=None, params=None, data=None, json_data=None):
    """
    Log HTTP request details.
    
    Args:
        url: Request URL
        method: HTTP method
        headers: Request headers
        params: URL parameters
        data: Request body data
        json_data: JSON request body
    """
    try:
        log_data = {
            "url": url,
            "method": method,
            "headers": headers,
            "params": params,
            "data": data,
            "json": json_data
        }
        logger.debug(f"HTTP Request: {json.dumps(log_data, default=str, indent=2)}")
    except Exception as e:
        logger.error(f"Error logging request: {e}")

def log_response(response):
    """
    Log HTTP response details.
    
    Args:
        response: Response object
    """
    try:
        log_data = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "content_length": len(response.content),
            "content_type": response.headers.get("Content-Type", ""),
            "elapsed": response.elapsed.total_seconds()
        }
        
        # Try to log response content if it's JSON
        if "application/json" in response.headers.get("Content-Type", ""):
            try:
                log_data["content"] = response.json()
            except Exception:
                # If JSON parsing fails, log the raw content (limited to 1000 chars)
                content = response.text[:1000]
                if len(response.text) > 1000:
                    content += "... [truncated]"
                log_data["content"] = content
        
        logger.debug(f"HTTP Response: {json.dumps(log_data, default=str, indent=2)}")
    except Exception as e:
        logger.error(f"Error logging response: {e}")

def log_exception(e, context=""):
    """
    Log exception details with traceback.
    
    Args:
        e: Exception object
        context: Additional context information
    """
    try:
        error_type = type(e).__name__
        error_msg = str(e)
        tb = traceback.format_exc()
        
        logger.error(f"Exception in {context}: {error_type}: {error_msg}")
        logger.debug(f"Traceback:\n{tb}")
    except Exception as log_error:
        logger.error(f"Error logging exception: {log_error}")

def log_data(data, label="Data"):
    """
    Log data object for debugging.
    
    Args:
        data: Data to log
        label: Label for the data
    """
    try:
        if isinstance(data, (dict, list)):
            logger.debug(f"{label}: {json.dumps(data, default=str, indent=2)}")
        else:
            logger.debug(f"{label}: {data}")
    except Exception as e:
        logger.error(f"Error logging data: {e}")
