"""
Ollama client for interacting with locally running LLMs.
"""
import json
import requests
from typing import Dict, List, Optional, Union, Any
from .logger import logger, log_request, log_response, log_exception, log_data

class OllamaClient:
    """Client for interacting with Ollama API to run local LLMs."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama client.

        Args:
            base_url: Base URL for the Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api"

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models in Ollama.

        Returns:
            List of model information dictionaries
        """
        url = f"{self.api_url}/tags"
        logger.info(f"Listing Ollama models from {url}")

        try:
            log_request(url, "GET")
            response = requests.get(url)
            log_response(response)

            response.raise_for_status()
            models = response.json().get("models", [])
            logger.debug(f"Found {len(models)} models")
            return models

        except requests.RequestException as e:
            log_exception(e, "list_models")
            logger.error(f"Error listing models: {e}")
            return []
        except json.JSONDecodeError as e:
            log_exception(e, "list_models - JSON parsing")
            logger.error(f"Error parsing JSON response: {e}")
            # Log the raw response content for debugging
            if 'response' in locals():
                logger.debug(f"Raw response content: {response.text[:1000]}")
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate a response from the model.

        Args:
            model: Name of the model to use (e.g., "llama2", "mistral")
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response

        Raises:
            ValueError: If model or prompt is invalid
            requests.RequestException: If there's an error communicating with Ollama
        """
        logger.info(f"Generating response using model: {model}")

        # Validate inputs
        if not model or not model.strip():
            logger.error("Invalid model name provided")
            raise ValueError("Invalid model name")

        if not prompt or not prompt.strip():
            logger.error("Empty prompt provided")
            raise ValueError("Empty prompt provided")

        # Sanitize inputs
        model = model.strip()
        prompt = prompt.strip()
        logger.debug(f"Prompt length: {len(prompt)} characters")

        # Log a truncated version of the prompt for debugging
        if len(prompt) > 500:
            logger.debug(f"Prompt (truncated): {prompt[:500]}...")
        else:
            logger.debug(f"Prompt: {prompt}")

        # Prepare the payload
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": max(0.0, min(1.0, temperature)),  # Ensure temperature is between 0 and 1
            "max_tokens": max(1, min(8192, max_tokens)),  # Reasonable limits for max_tokens
            "stream": True  # Always use streaming for more reliable responses
        }

        if system_prompt:
            payload["system"] = system_prompt.strip()
            logger.debug(f"Using system prompt: {system_prompt[:200]}...")

        url = f"{self.api_url}/generate"

        try:
            # First check if Ollama is available
            logger.debug("Checking Ollama availability")
            if not self.check_availability():
                logger.error("Ollama service is not available")
                raise ConnectionError("Ollama service is not available. Please make sure it's running.")

            # Check if the model exists
            logger.debug("Checking if model exists")
            models = self.list_models()
            # Get both full model names and base names
            full_model_names = [m.get("name", "") for m in models]
            base_model_names = [name.split(":")[0] for name in full_model_names]

            logger.debug(f"Available models (full names): {full_model_names}")
            logger.debug(f"Available models (base names): {base_model_names}")

            # Check if the model exists in any form
            model_exists = (
                model in full_model_names or  # Exact match (e.g., "deepseek-r1:7b")
                model in base_model_names or  # Base name match (e.g., "deepseek-r1")
                f"{model}:latest" in full_model_names  # With :latest suffix
            )

            if not model_exists:
                # If model doesn't exist, try to find a match with the base name
                if ":" in model:  # If model has a tag (e.g., "deepseek-r1:7b")
                    base_name = model.split(":")[0]
                    if base_name in base_model_names:
                        # Use the first matching model with this base name
                        matching_models = [m for m in full_model_names if m.startswith(base_name + ":")]
                        if matching_models:
                            logger.info(f"Using model '{matching_models[0]}' instead of '{model}'")
                            model = matching_models[0]
                            model_exists = True

                if not model_exists:
                    logger.error(f"Model '{model}' not found in Ollama")
                    raise ValueError(f"Model '{model}' not found in Ollama. Available models: {', '.join(full_model_names)}")

            # Make the request with a timeout
            logger.debug(f"Sending request to {url}")
            log_request(url, "POST", json_data=payload)

            # Set stream=True to handle streaming responses properly
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=120  # Increased timeout for longer generations
            )

            # Handle HTTP errors
            response.raise_for_status()

            # Process the streaming response
            logger.debug("Processing streaming response from Ollama")
            full_response = ""

            # Collect the response pieces
            for line in response.iter_lines():
                if not line:
                    continue

                # Decode the line
                line_str = line.decode('utf-8')

                try:
                    # Parse the JSON object
                    chunk = json.loads(line_str)

                    # Check for errors
                    if "error" in chunk:
                        error_msg = chunk.get('error', 'Unknown error')
                        logger.error(f"Ollama API error: {error_msg}")
                        raise ValueError(f"Ollama API error: {error_msg}")

                    # Extract the response piece
                    if "response" in chunk:
                        response_piece = chunk.get("response", "")
                        full_response += response_piece

                    # Check if this is the final chunk
                    if chunk.get("done", False):
                        logger.debug("Received final chunk with done=true")
                        break

                except json.JSONDecodeError as e:
                    # Log the error but continue processing
                    logger.warning(f"Could not parse line as JSON: {e}")
                    logger.debug(f"Problematic line: {line_str}")
                    # Continue processing other lines

            # Check if we got any response
            if not full_response:
                logger.warning("No response content received from Ollama")
                raise ValueError("No response content received from Ollama")

            logger.debug(f"Generated full response length: {len(full_response)} characters")

            # Log a truncated version of the result
            if len(full_response) > 500:
                logger.debug(f"Generated response (truncated): {full_response[:500]}...")
            else:
                logger.debug(f"Generated response: {full_response}")

            return full_response

        except requests.RequestException as e:
            log_exception(e, "generate - HTTP request")
            logger.error(f"Error generating response: {e}")
            raise ValueError(f"Error communicating with Ollama: {str(e)}")
        except json.JSONDecodeError as e:
            log_exception(e, "generate - JSON parsing")
            logger.error(f"JSON parsing error: {e}")
            raise ValueError(f"Error parsing Ollama response: {str(e)}")
        except ValueError as e:
            log_exception(e, "generate - Value error")
            logger.error(f"Value error: {e}")
            raise  # Re-raise the exception to be handled by the caller
        except Exception as e:
            log_exception(e, "generate - Unexpected error")
            logger.error(f"Unexpected error in generate: {e}")
            raise ValueError(f"Error generating response: {str(e)}")

    def check_availability(self) -> bool:
        """
        Check if Ollama is running and available.

        Returns:
            True if Ollama is available, False otherwise
        """
        url = f"{self.api_url}/tags"
        logger.debug(f"Checking Ollama availability at {url}")

        try:
            log_request(url, "GET")
            response = requests.get(url, timeout=5)  # Short timeout for availability check
            log_response(response)

            available = response.status_code == 200
            logger.debug(f"Ollama available: {available}")
            return available

        except requests.RequestException as e:
            log_exception(e, "check_availability")
            logger.debug(f"Ollama not available: {e}")
            return False

    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama library.

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful, False otherwise
        """
        url = f"{self.api_url}/pull"
        logger.info(f"Pulling model {model_name} from Ollama")

        try:
            payload = {"name": model_name}
            log_request(url, "POST", json_data=payload)

            response = requests.post(
                url,
                json=payload,
                timeout=300  # Longer timeout for model pulling
            )

            log_response(response)
            response.raise_for_status()

            logger.info(f"Successfully pulled model {model_name}")
            return True

        except requests.RequestException as e:
            log_exception(e, f"pull_model - {model_name}")
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
