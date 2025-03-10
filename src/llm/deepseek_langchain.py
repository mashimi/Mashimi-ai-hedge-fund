"""
DeepSeek API integration for LangChain.
"""

from typing import Any, Dict, List, Mapping, Optional, Union
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
import requests
import json


class ChatDeepSeek(BaseChatModel):
    """DeepSeek chat model."""

    model: str = "deepseek-chat"
    """Model name to use."""
    api_key: Optional[str] = None
    """DeepSeek API key."""
    api_base: str = "https://api.deepseek.com/v1"
    """Base URL for DeepSeek API."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""
    streaming: bool = False
    """Whether to stream the results."""

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "deepseek-chat"

    def _convert_messages_to_deepseek_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, str]]:
        """Convert messages to the format expected by DeepSeek API."""
        deepseek_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                deepseek_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                deepseek_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                deepseek_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, ChatMessage):
                role = message.role
                if role == "assistant":
                    deepseek_messages.append({"role": "assistant", "content": message.content})
                elif role == "user":
                    deepseek_messages.append({"role": "user", "content": message.content})
                elif role == "system":
                    deepseek_messages.append({"role": "system", "content": message.content})
                else:
                    deepseek_messages.append({"role": "user", "content": message.content})
            else:
                raise ValueError(f"Message type {type(message)} not supported")
        return deepseek_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response."""
        if self.api_key is None:
            raise ValueError("DeepSeek API key not provided")

        deepseek_messages = self._convert_messages_to_deepseek_format(messages)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": deepseek_messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            data["top_p"] = self.top_p
        if stop is not None:
            data["stop"] = stop
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code != 200:
            raise ValueError(
                f"Error from DeepSeek API: {response.status_code} {response.text}"
            )
        
        response_json = response.json()
        
        message_content = response_json["choices"][0]["message"]["content"]
        
        message = AIMessage(content=message_content)
        
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"token_usage": response_json.get("usage", {})}
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generate a chat response."""
        # For simplicity, we're using the synchronous implementation
        # In a real implementation, you would use aiohttp or similar
        return self._generate(messages, stop, run_manager, **kwargs)
