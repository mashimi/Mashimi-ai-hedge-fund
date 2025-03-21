import os
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from enum import Enum
from pydantic import BaseModel
from typing import Tuple
from llm.perplexity_langchain import ChatPerplexity
from llm.deepseek_langchain import ChatDeepSeek


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""
    OPENAI = "OpenAI"
    GROQ = "Groq"
    ANTHROPIC = "Anthropic"
    PERPLEXITY = "Perplexity"  # Added Perplexity as a provider
    DEEPSEEK = "DeepSeek"  # Added DeepSeek as a provider


class LLMModel(BaseModel):
    """Represents an LLM model configuration"""
    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)
    
    def is_deepseek(self) -> bool:
        """Check if the model is a DeepSeek model"""
        return self.model_name.startswith("deepseek")

    def is_perplexity(self) -> bool:
        """Check if the model is a Perplexity model"""
        return self.provider == ModelProvider.PERPLEXITY


# Define available models including new Perplexity Sonar models and DeepSeek models
AVAILABLE_MODELS = [
    LLMModel(
        display_name="[anthropic] claude-3.5-haiku",
        model_name="claude-3-5-haiku-latest",
        provider=ModelProvider.ANTHROPIC
    ),
    LLMModel(
        display_name="[anthropic] claude-3.5-sonnet",
        model_name="claude-3-5-sonnet-latest",
        provider=ModelProvider.ANTHROPIC
    ),
    LLMModel(
        display_name="[anthropic] claude-3.7-sonnet",
        model_name="claude-3-7-sonnet-latest",
        provider=ModelProvider.ANTHROPIC
    ),
    LLMModel(
        display_name="[groq] deepseek-r1 70b",
        model_name="deepseek-r1-distill-llama-70b",
        provider=ModelProvider.GROQ
    ),
    LLMModel(
        display_name="[groq] llama-3.3 70b",
        model_name="llama-3.3-70b-versatile",
        provider=ModelProvider.GROQ
    ),
    LLMModel(
        display_name="[openai] gpt-4o",
        model_name="gpt-4o",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[openai] gpt-4o-mini",
        model_name="gpt-4o-mini",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[openai] o1",
        model_name="o1",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[openai] o3-mini",
        model_name="o3-mini",
        provider=ModelProvider.OPENAI
    ),
    # DeepSeek models
    LLMModel(
        display_name="[deepseek] deepseek-chat",
        model_name="deepseek-chat",
        provider=ModelProvider.DEEPSEEK
    ),
    LLMModel(
        display_name="[deepseek] deepseek-coder",
        model_name="deepseek-coder",
        provider=ModelProvider.DEEPSEEK
    ),
    # New Perplexity Sonar models
    LLMModel(
        display_name="[perplexity] sonar-small-online",
        model_name="sonar-small-online",
        provider=ModelProvider.PERPLEXITY
    ),
    LLMModel(
        display_name="[perplexity] sonar-medium-online",
        model_name="sonar-medium-online",
        provider=ModelProvider.PERPLEXITY
    ),
    LLMModel(
        display_name="[perplexity] sonar-large-online",
        model_name="sonar-large-online",
        provider=ModelProvider.PERPLEXITY
    ),
    LLMModel(
        display_name="[perplexity] sonar-reasoning",
        model_name="sonar-reasoning",
        provider=ModelProvider.PERPLEXITY
    ),
]

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

def get_model_info(model_name: str) -> LLMModel | None:
    """Get model information by model_name"""
    return next((model for model in AVAILABLE_MODELS if model.model_name == model_name), None)

def get_model(model_name: str, model_provider: ModelProvider) -> ChatOpenAI | ChatGroq | ChatAnthropic | ChatPerplexity | ChatDeepSeek | None:
    """Get the appropriate LLM model based on provider and model name"""
    if model_provider == ModelProvider.GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure GROQ_API_KEY is set in your .env file.")
            raise ValueError("Groq API key not found. Please make sure GROQ_API_KEY is set in your .env file.")
        return ChatGroq(model=model_name, api_key=api_key)
    
    elif model_provider == ModelProvider.OPENAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file.")
            raise ValueError("OpenAI API key not found. Please make sure OPENAI_API_KEY is set in your .env file.")
        return ChatOpenAI(model=model_name, api_key=api_key)
    
    elif model_provider == ModelProvider.ANTHROPIC:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure ANTHROPIC_API_KEY is set in your .env file.")
            raise ValueError("Anthropic API key not found. Please make sure ANTHROPIC_API_KEY is set in your .env file.")
        return ChatAnthropic(model=model_name, api_key=api_key)
    
    elif model_provider == ModelProvider.PERPLEXITY:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure PERPLEXITY_API_KEY is set in your .env file.")
            raise ValueError("Perplexity API key not found. Please make sure PERPLEXITY_API_KEY is set in your .env file.")
        return ChatPerplexity(model=model_name, api_key=api_key)
    
    elif model_provider == ModelProvider.DEEPSEEK:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure DEEPSEEK_API_KEY is set in your .env file.")
            raise ValueError("DeepSeek API key not found. Please make sure DEEPSEEK_API_KEY is set in your .env file.")
        return ChatDeepSeek(model=model_name, api_key=api_key)
    
    return None
