# DeepSeek API Integration

This document explains how to integrate the DeepSeek API with the AI Hedge Fund project.

## Overview

DeepSeek provides powerful language models that can be used for financial analysis and trading strategy development. This integration allows you to use DeepSeek models alongside other LLM providers in the AI Hedge Fund system.

## Setup

1. **Get API Key**:
   - Visit [DeepSeek](https://platform.deepseek.com) and create an account
   - Navigate to the API section and generate an API key
   - Copy your API key for the next step

2. **Configure Environment**:
   - Add your DeepSeek API key to the `.env` file:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   ```

3. **Available Models**:
   - `deepseek-chat`: General-purpose chat model
   - `deepseek-coder`: Specialized for code generation and analysis

## Usage

### Command Line

When running the hedge fund from the command line, you can specify a DeepSeek model:

```bash
python src/main.py --tickers AAPL,MSFT,GOOGL --model-name deepseek-chat --model-provider DeepSeek
```

### Dashboard

When using the Streamlit dashboard, you can select DeepSeek models from the dropdown menu in the sidebar.

## Implementation Details

The DeepSeek integration consists of:

1. **LangChain Integration** (`src/llm/deepseek_langchain.py`):
   - Implements the `ChatDeepSeek` class that extends LangChain's `BaseChatModel`
   - Handles message formatting and API communication

2. **Model Registry** (`src/llm/models.py`):
   - Adds DeepSeek as a supported model provider
   - Registers available DeepSeek models
   - Provides factory methods to instantiate DeepSeek models

## Troubleshooting

- **API Key Errors**: Ensure your DeepSeek API key is correctly set in the `.env` file
- **Model Not Found**: Verify you're using one of the supported model names (`deepseek-chat` or `deepseek-coder`)
- **Rate Limiting**: DeepSeek may have rate limits on API calls; check their documentation for details

## References

- [DeepSeek API Documentation](https://platform.deepseek.com/docs)
- [DeepSeek Model Capabilities](https://platform.deepseek.com/models)
