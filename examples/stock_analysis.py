"""
Example script demonstrating how to use the StockPredictor class.
"""
import os
import sys
import json
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.stock_predictor import StockPredictor
from src.tools.api import get_financial_metrics, get_price_data, get_market_cap

def main():
    """Run the stock analysis example."""
    # Load environment variables
    load_dotenv()
    
    # Get ticker from command line arguments or use default
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Analyzing {ticker}...")
    
    # Initialize the StockPredictor
    predictor = StockPredictor(model_type='reasoning', temperature=0.7)
    
    # Get financial data
    print(f"Fetching financial data for {ticker}...")
    try:
        financial_metrics = get_financial_metrics(ticker, period="ttm", limit=5)
        market_cap = get_market_cap(ticker)
    except Exception as e:
        error_msg = str(e)
        print(f"Error fetching financial data: {error_msg}")
        
        # Check for insufficient credits error
        if "Insufficient credits" in error_msg:
            print("\nNOTE: The Financial Datasets API requires credits for most tickers.")
            print("Free tickers include: AAPL, GOOGL, MSFT, NVDA, and TSLA.")
            print("For other tickers, you need to purchase credits at https://financialdatasets.ai/")
            
            # Ask if the user wants to continue with limited data
            continue_analysis = input("\nDo you want to continue analysis with limited data? (y/n): ").lower().strip() == 'y'
            if not continue_analysis:
                print("Analysis cancelled.")
                sys.exit(0)
        
        financial_metrics = []
        market_cap = None
    
    # Get price data
    print(f"Fetching price data for {ticker}...")
    try:
        price_data = get_price_data(ticker, limit=30)
    except Exception as e:
        error_msg = str(e)
        print(f"Error fetching price data: {error_msg}")
        
        # Check for insufficient credits error
        if "Insufficient credits" in error_msg and not 'continue_analysis' in locals():
            print("\nNOTE: The Financial Datasets API requires credits for most tickers.")
            print("Free tickers include: AAPL, GOOGL, MSFT, NVDA, and TSLA.")
            print("For other tickers, you need to purchase credits at https://financialdatasets.ai/")
            
            # Ask if the user wants to continue with limited data
            continue_analysis = input("\nDo you want to continue analysis with limited data? (y/n): ").lower().strip() == 'y'
            if not continue_analysis:
                print("Analysis cancelled.")
                sys.exit(0)
        
        price_data = None
    
    # Prepare data for analysis
    data = {
        "financial_data": {
            "metrics": [metric.model_dump() for metric in financial_metrics],
            "market_cap": market_cap
        },
        "price_data": price_data.to_dict() if hasattr(price_data, 'to_dict') else {},
        "technical_indicators": {
            "moving_averages": {
                "sma_50": 150.25,  # Example values
                "sma_200": 145.75,
                "ema_12": 152.30,
                "ema_26": 148.50
            },
            "oscillators": {
                "rsi_14": 65.5,
                "macd": 2.5,
                "macd_signal": 1.8,
                "macd_histogram": 0.7
            }
        },
        "sentiment_data": {
            "news_sentiment": {
                "positive": 65,
                "negative": 25,
                "neutral": 10
            },
            "social_media": {
                "twitter_sentiment": 0.65,
                "reddit_sentiment": 0.58,
                "stocktwits_sentiment": 0.72
            },
            "analyst_ratings": {
                "buy": 25,
                "hold": 10,
                "sell": 5
            }
        }
    }
    
    # Analyze the ticker
    print(f"Performing comprehensive analysis of {ticker}...")
    analysis = predictor.analyze_ticker(ticker, data)
    
    # Print the results
    print("\n=== ANALYSIS RESULTS ===\n")
    
    # Print recommendation
    recommendation = json.loads(analysis["recommendation"])
    print(f"RECOMMENDATION FOR {ticker}:")
    print(f"Signal: {recommendation['recommendation']['signal'].upper()}")
    print(f"Confidence: {recommendation['recommendation']['confidence']}%")
    print(f"Time Horizon: {recommendation['recommendation']['time_horizon']}")
    print(f"\nReasoning: {recommendation['recommendation']['reasoning']}")
    
    print("\n=== SUMMARY ===")
    print(recommendation['summary'])
    
    print("\n=== RISKS ===")
    print(recommendation['risks'])
    
    # Save the full analysis to a file
    output_file = f"{ticker}_analysis.json"
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nFull analysis saved to {output_file}")

if __name__ == "__main__":
    main()
