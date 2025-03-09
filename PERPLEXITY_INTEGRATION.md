# Perplexity Sonar Reasoning Integration

This document describes the integration of Perplexity's Sonar Reasoning model into the AI Hedge Fund project.

## Overview

Perplexity's Sonar Reasoning model has been integrated into the AI Hedge Fund project to provide enhanced stock analysis capabilities. The Sonar Reasoning model is specifically designed for complex reasoning tasks and can provide detailed analysis of stocks based on fundamental, technical, and sentiment data.

## Features

- **Perplexity Sonar Models**: Support for multiple Perplexity models (sonar-reasoning, sonar-small-online, sonar-medium-online, sonar-large-online)
- **Enhanced Analysis**: Improved reasoning capabilities for stock analysis
- **StockPredictor Class**: A new class for comprehensive stock analysis using Perplexity models
- **Command-Line Interface**: A new script for analyzing any ticker from the command line
- **Dashboard Integration**: Perplexity models available in the Streamlit dashboard

## Components

### 1. Perplexity API Client

The `PerplexityClient` class in `src/llm/perplexity.py` provides a client for accessing the Perplexity API. It supports:

- Authentication with API key
- Customizable model parameters (max_tokens, temperature)
- Error handling and response parsing

### 2. LangChain Integration

The `ChatPerplexity` class in `src/llm/perplexity_langchain.py` integrates Perplexity models with LangChain, allowing:

- Seamless use of Perplexity models in LangChain workflows
- Conversion between LangChain and Perplexity message formats
- Support for all LangChain features (callbacks, streaming, etc.)

### 3. Model Configuration

The `src/llm/models.py` file has been updated to include Perplexity models in the available models list, enabling:

- Selection of Perplexity models in the UI
- Automatic API key validation
- Consistent model interface across providers

### 4. StockPredictor Class

The new `StockPredictor` class in `src/utils/stock_predictor.py` provides a powerful interface for analyzing stocks using Perplexity's Sonar Reasoning model:

- Comprehensive stock analysis (fundamental, technical, sentiment)
- Customizable model parameters
- Detailed JSON output with investment recommendations

### 5. Command-Line Interface

The `analyze_ticker.py` script provides a command-line interface for analyzing any ticker:

```bash
./analyze_ticker.py --ticker AAPL --model reasoning --temperature 0.7
```

See [examples/README.md](examples/README.md) for more details.

## Setup

1. **Install Dependencies**:
   ```bash
   poetry install
   ```

2. **Set API Keys**:
   Add your API keys to the `.env` file:
   ```
   # Required for Perplexity models
   PERPLEXITY_API_KEY=your-perplexity-api-key
   
   # Required for financial data beyond free tickers
   FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
   ```

3. **Run the Dashboard**:
   ```bash
   poetry run streamlit run dashboard.py
   ```
   Select a Perplexity model from the dropdown menu.

4. **Analyze a Ticker**:
   ```bash
   # Analyze a free ticker (AAPL, GOOGL, MSFT, NVDA, TSLA)
   ./analyze_ticker.py --ticker AAPL
   
   # Analyze any ticker (requires Financial Datasets API credits)
   ./analyze_ticker.py --ticker TBL
   ```

### Note on Financial Data

The Financial Datasets API provides free data for a limited set of tickers:
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- NVDA (NVIDIA)
- TSLA (Tesla)

For other tickers, you need to purchase credits at [Financial Datasets](https://financialdatasets.ai/). The scripts will handle the "Insufficient credits" error gracefully and allow you to continue analysis with limited data.

## Example Usage

### Using StockPredictor in Code

```python
from src.utils.stock_predictor import StockPredictor

# Initialize the predictor
predictor = StockPredictor(model_type='reasoning', temperature=0.7)

# Analyze a ticker
analysis = predictor.analyze_ticker('AAPL', data)

# Get the recommendation
recommendation = json.loads(analysis["recommendation"])
print(f"Signal: {recommendation['recommendation']['signal']}")
print(f"Confidence: {recommendation['recommendation']['confidence']}%")
```

### Using the Command-Line Interface

```bash
# Basic usage
./analyze_ticker.py --ticker AAPL

# Specify model and parameters
./analyze_ticker.py --ticker MSFT --model reasoning --temperature 0.7

# Print detailed analysis
./analyze_ticker.py --ticker NVDA --verbose
```

### Using the Dashboard

1. Run the dashboard: `poetry run streamlit run dashboard.py`
2. Enter a ticker symbol (e.g., AAPL)
3. Select a Perplexity model from the dropdown menu
4. Click "Run Analysis"

## Benefits

- **Enhanced Reasoning**: Perplexity's Sonar Reasoning model provides more detailed and nuanced analysis
- **Flexibility**: Support for multiple models with different capabilities and performance characteristics
- **Ease of Use**: Simple interfaces for using Perplexity models in various contexts
- **Integration**: Seamless integration with existing AI Hedge Fund components
