# AI Hedge Fund

An AI-powered hedge fund that uses multiple AI agents to make trading decisions.

## Getting Started

### Prerequisites

- Python 3.10+
- Financial Datasets API key (set in .env file)

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API key:

```
FINANCIAL_DATASETS_API_KEY=your_api_key_here
```

## Usage

### Running the Hedge Fund

```bash
python src/main.py --tickers AAPL,MSFT,GOOGL --start-date 2023-01-01 --end-date 2023-04-01
```

### Command Line Arguments

- `--tickers`: Comma-separated list of stock ticker symbols (required)
- `--start-date`: Start date in YYYY-MM-DD format (defaults to 3 months before end date)
- `--end-date`: End date in YYYY-MM-DD format (defaults to today)
- `--initial-cash`: Initial cash position (defaults to 100,000)
- `--margin-requirement`: Initial margin requirement (defaults to 0)
- `--show-reasoning`: Show detailed reasoning from each agent
- `--show-agent-graph`: Show the agent workflow graph

### Using the Dashboard

For a more interactive experience, you can use the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

## Features

### AI Analysts

The hedge fund includes multiple AI analysts with different investment philosophies:

- **Investment Legends**
  - Warren Buffett
  - Charlie Munger
  - Ben Graham
  - Bill Ackman
  - Cathie Wood

- **Analysis Styles**
  - Fundamentals Analyst
  - Technical Analyst
  - Sentiment Analyst
  - Valuation Analyst

- **Trading Strategies**
  - Momentum Strategy: Technical analysis with entry/exit points and stop-loss levels
  - Value Strategy: Fundamental analysis with portfolio allocation recommendations

### How It Works

1. **Data Collection**: The system collects financial data for the specified tickers
2. **Multi-Agent Analysis**: Each AI agent analyzes the data from its unique perspective
3. **Risk Management**: A risk management agent evaluates all signals
4. **Portfolio Management**: The portfolio manager makes the final trading decisions
5. **Strategy Integration**: Trading strategies provide specific entry/exit points and allocation recommendations

## Getting Price Data for Momentum Strategy

The momentum strategy requires historical price data to calculate technical indicators. Follow these steps:

1. **Obtain API Access**:
   - Visit [Financial Datasets](https://www.financialdatasets.com) to get an API key
   - Select the "Historical Prices" package
   - Copy your API key

2. **Configure Environment**:
   ```bash
   echo "FINANCIAL_DATASETS_API_KEY=your_api_key_here" > .env
   ```

3. **Verify Data Availability**:
   ```bash
   python -c "from src.tools.api import get_prices; print(get_prices('AAPL', '2023-01-01', '2023-04-01'))"
   ```

4. **Troubleshooting Missing Data**:
   - Check supported tickers: Most major US stocks are supported
   - Validate date ranges: Data available from 2000-01-01 to present
   - Ensure sufficient history: Minimum 180 days required for momentum analysis
   - If you see "No price data found" errors, the system will automatically provide a neutral signal with detailed reasoning

## Troubleshooting

### Common Issues

- **API Key Errors**: Ensure your Financial Datasets API key is correctly set in the `.env` file
- **Missing Data**: Some tickers may not have complete data available
- **Date Range Issues**: Ensure your date range is valid and not too far in the past

### Error Messages

- **"No price data found"**: The API couldn't find price data for the specified ticker and date range
- **"Invalid line items"**: The API doesn't support some of the requested financial line items
- **"Unable to fetch financial data"**: There was an error fetching data from the API

## License

This project is licensed under the MIT License - see the LICENSE file for details.
