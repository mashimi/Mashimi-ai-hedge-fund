"""
Perplexity Sonar Reasoning model client for the AI Hedge Fund.
"""
import os
import requests
import json
from typing import Dict, Any, Optional


class PerplexityClient:
    """Client for accessing Perplexity Sonar Reasoning model."""
    
    API_URL = "https://api.perplexity.ai/chat/completions"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Perplexity client with API key."""
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("Perplexity API key not found. Please set PERPLEXITY_API_KEY in your .env file.")
    
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
        
        response = requests.post(
            self.API_URL,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Error from Perplexity API: {response.status_code} - {response.text}")
        
        return response.json()
