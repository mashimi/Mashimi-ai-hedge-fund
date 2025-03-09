# Stock Analysis with Perplexity Sonar Reasoning

This directory contains examples of how to use the Perplexity Sonar Reasoning model for stock analysis.

## StockPredictor Class

The `StockPredictor` class in `src/utils/stock_predictor.py` provides a powerful interface for analyzing stocks using Perplexity's Sonar Reasoning model. This model is specifically designed for complex reasoning tasks and can provide detailed analysis of stocks based on fundamental, technical, and sentiment data.

### Features

- Comprehensive stock analysis using Perplexity Sonar Reasoning
- Support for fundamental, technical, and sentiment analysis
- Customizable model parameters (temperature, max tokens)
- Multiple Perplexity models supported (reasoning, small, medium, large)
- Detailed JSON output with investment recommendations

## Example Scripts

### stock_analysis.py

This script demonstrates how to use the `StockPredictor` class to analyze a stock ticker. It fetches financial data, price data, and market cap for the specified ticker, then performs a comprehensive analysis using the Perplexity Sonar Reasoning model.

#### Usage

```bash
# Analyze AAPL (default)
python stock_analysis.py

# Analyze a specific ticker
python stock_analysis.py MSFT
```

### Command-Line Interface (analyze_ticker.py)

The `analyze_ticker.py` script in the root directory provides a command-line interface for analyzing stock tickers using the Perplexity Sonar Reasoning model.

#### Usage

```bash
# Basic usage
./analyze_ticker.py --ticker AAPL

# Specify model and parameters
./analyze_ticker.py --ticker MSFT --model reasoning --temperature 0.7 --max-tokens 4096

# Save output to a specific file
./analyze_ticker.py --ticker GOOGL --output google_analysis.json

# Print detailed analysis
./analyze_ticker.py --ticker NVDA --verbose
```

#### Options

- `--ticker`: Stock ticker symbol to analyze (required)
- `--model`: Perplexity model to use (choices: reasoning, small, medium, large; default: reasoning)
- `--temperature`: Temperature for sampling (0.0 to 1.0; default: 0.7)
- `--max-tokens`: Maximum number of tokens to generate (default: model-specific)
- `--output`: Output file path for saving the analysis (default: ticker_analysis.json)
- `--verbose`: Print detailed analysis

## Requirements

- Python 3.9+
- Perplexity API key (set as PERPLEXITY_API_KEY in .env file)
- Financial Datasets API key (set as FINANCIAL_DATASETS_API_KEY in .env file) for non-free tickers
- Dependencies from pyproject.toml

### Note on Financial Data

The Financial Datasets API provides free data for a limited set of tickers:
- AAPL (Apple)
- GOOGL (Google)
- MSFT (Microsoft)
- NVDA (NVIDIA)
- TSLA (Tesla)

For other tickers, you need to purchase credits at [Financial Datasets](https://financialdatasets.ai/). The scripts will handle the "Insufficient credits" error gracefully and allow you to continue analysis with limited data.

## Example Output

The analysis output includes:

- Fundamental analysis (financial health, growth, competitive position, valuation)
- Technical analysis (trend, support/resistance, momentum, volume)
- Sentiment analysis (news, social media, analyst opinions, institutional activity)
- Final recommendation with confidence level and time horizon
- Summary of key points, areas of agreement/conflict, and risks

The output is saved as a JSON file for further processing or visualization.
