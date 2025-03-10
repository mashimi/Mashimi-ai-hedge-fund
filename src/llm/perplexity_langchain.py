"""
LangChain integration for Perplexity Sonar Reasoning.
"""
import json
import logging
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from llm.perplexity import PerplexityClient, PerplexityAPIError

# Set up logging
logger = logging.getLogger(__name__)


class ChatPerplexity(BaseChatModel):
    """LangChain integration for the Perplexity Sonar Reasoning models."""
    
    client: Optional[PerplexityClient] = None
    model_name: str = "sonar-medium-online"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0
    
    def __init__(
        self, 
        model: str = "sonar-medium-online",
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
        pplx_api_key: Optional[str] = None,  # For backward compatibility
        **kwargs
    ):
        """Initialize with model and API key."""
        super().__init__(**kwargs)
        # Use pplx_api_key if provided (for backward compatibility)
        effective_api_key = pplx_api_key or api_key
        self.client = PerplexityClient(
            api_key=effective_api_key,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout
        )
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "perplexity-sonar"
    
    def _convert_messages_to_perplexity_format(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to Perplexity format."""
        perplexity_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                perplexity_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                perplexity_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                perplexity_messages.append({"role": "system", "content": message.content})
            else:
                raise ValueError(f"Message type {type(message)} not supported")
        return perplexity_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from Perplexity Sonar."""
        perplexity_messages = self._convert_messages_to_perplexity_format(messages)
        
        try:
            # Call the Perplexity API
            response = self.client.completion(
                messages=perplexity_messages,
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract the response text
            response_text = response["choices"][0]["message"]["content"]
            
        except PerplexityAPIError as e:
            # Log the error
            logger.error(f"Perplexity API Error: {str(e)}")
            
            # Use fallback response
            fallback_response = self.client.get_fallback_response(self.model_name)
            response_text = fallback_response["choices"][0]["message"]["content"]
            
            # If this is a system message, try to extract a valid JSON
            if any(isinstance(m, SystemMessage) for m in messages):
                try:
                    # Check if the response is already valid JSON
                    json.loads(response_text)
                except json.JSONDecodeError:
                    # If not, create a simple JSON response
                    response_text = json.dumps({
                        "error": f"API Error: {str(e)}",
                        "message": "The Perplexity API is currently unavailable. Please try again later."
                    })
        
        # Create a LangChain chat generation object
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
