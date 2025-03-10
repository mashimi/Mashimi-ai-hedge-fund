"""
Perplexity Sonar Reasoning model client for the AI Hedge Fund.
"""
import os
import requests
import json
import time
from typing import Dict, Any, Optional


class PerplexityAPIError(Exception):
    """Exception raised for errors in the Perplexity API."""
    
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Perplexity API Error ({status_code}): {message}")


class PerplexityClient:
    """Client for accessing Perplexity Sonar Reasoning model."""
    
    API_URL = "https://api.perplexity.ai/chat/completions"
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        max_retries: int = 3, 
        retry_delay: float = 1.0,
        timeout: float = 60.0
    ):
        """
        Initialize the Perplexity client with API key.
        
        Args:
            api_key: Perplexity API key
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            timeout: Timeout for API requests in seconds
        """
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("Perplexity API key not found. Please set PERPLEXITY_API_KEY in your .env file.")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
    
    def completion(
        self, 
        messages: list[Dict[str, Any]], 
        model: str = "sonar-medium-online",
        max_tokens: int = None,
        temperature: float = None
    ) -> Dict[str, Any]:
        """
        Call the Perplexity API with messages.
        
        Args:
            messages: List of message dictionaries (role + content)
            model: Perplexity model to use (default: sonar-medium-online)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling (0.0 to 1.0)
            
        Returns:
            API response as dictionary
            
        Raises:
            PerplexityAPIError: If the API returns an error
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        # Add optional parameters if provided
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
            
        if temperature is not None:
            payload["temperature"] = temperature
        
        # Retry logic
        retries = 0
        last_error = None
        
        while retries <= self.max_retries:
            try:
                response = requests.post(
                    self.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout  # Use configurable timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                
                # Handle different error codes
                if response.status_code == 429:
                    # Rate limit exceeded, retry after delay
                    retry_after = int(response.headers.get("Retry-After", self.retry_delay))
                    time.sleep(retry_after)
                    retries += 1
                    continue
                
                if response.status_code >= 500:
                    # Server error, retry after delay
                    if retries < self.max_retries:
                        time.sleep(self.retry_delay * (2 ** retries))  # Exponential backoff
                        retries += 1
                        continue
                
                # For other errors, raise exception
                error_message = response.text
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_message = error_json["error"]
                except:
                    pass
                
                raise PerplexityAPIError(response.status_code, error_message)
                
            except requests.RequestException as e:
                # Network error, retry after delay
                if retries < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** retries))  # Exponential backoff
                    retries += 1
                    last_error = e
                    continue
                
                raise PerplexityAPIError(0, f"Network error: {str(e)}")
        
        # If we've exhausted retries, raise the last error
        if last_error:
            raise PerplexityAPIError(0, f"Max retries exceeded: {str(last_error)}")
        
        # This should never happen, but just in case
        raise PerplexityAPIError(0, "Unknown error occurred")
    
    def get_fallback_response(self, model: str) -> Dict[str, Any]:
        """
        Get a fallback response when the API is unavailable.
        
        Args:
            model: The model that was being used
            
        Returns:
            A fallback response dictionary
        """
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "summary": "Analysis could not be completed due to API unavailability.",
                            "agreements": "N/A",
                            "conflicts": "N/A",
                            "risks": "Unable to assess risks due to API unavailability.",
                            "recommendation": {
                                "signal": "neutral",
                                "confidence": 50,
                                "reasoning": "The Perplexity API is currently unavailable. Please try again later.",
                                "time_horizon": "medium term"
                            }
                        })
                    }
                }
            ]
        }
