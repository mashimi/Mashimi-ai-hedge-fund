"""
LangChain integration for Perplexity Sonar Reasoning.
"""
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

from llm.perplexity import PerplexityClient


class ChatPerplexity(BaseChatModel):
    """LangChain integration for the Perplexity Sonar Reasoning models."""
    
    client: Optional[PerplexityClient] = None
    model_name: str = "sonar-medium-online"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    
    def __init__(
        self, 
        model: str = "sonar-medium-online",
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        pplx_api_key: Optional[str] = None,  # For backward compatibility
        **kwargs
    ):
        """Initialize with model and API key."""
        super().__init__(**kwargs)
        # Use pplx_api_key if provided (for backward compatibility)
        effective_api_key = pplx_api_key or api_key
        self.client = PerplexityClient(api_key=effective_api_key)
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
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
        
        # Call the Perplexity API
        response = self.client.completion(
            messages=perplexity_messages,
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        # Extract the response text
        response_text = response["choices"][0]["message"]["content"]
        
        # Create a LangChain chat generation object
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
