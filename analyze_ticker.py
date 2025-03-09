#!/usr/bin/env python
"""
Command-line interface for analyzing stock tickers using Perplexity Sonar Reasoning.
"""
import os
import sys
import json
import argparse
from dotenv import load_dotenv

from src.utils.stock_predictor import StockPredictor, PerplexityConfig
from src.tools.api import get_financial_metrics, get_price_data, get_market_cap
from src.utils.display import print_colored

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze stock tickers using Perplexity Sonar Reasoning")
    
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol to analyze")
    
    parser.add_argument(
        "--model", 
        type=str, 
        choices=list(PerplexityConfig.MODELS.keys()),
        default="reasoning",
        help="Perplexity model to use for analysis"
    )
    
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for sampling (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=None,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output file path for saving the analysis (default: ticker_analysis.json)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print detailed analysis"
    )
    
    return parser.parse_args()

def main():
    """Run the stock analysis CLI."""
    # Load environment variables
    load_dotenv()
    
    # Parse command-line arguments
    args = parse_args()
    ticker = args.ticker.upper()
    
    # Check if API key is set
    api_key = os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY")
    if not api_key:
        print_colored("Error: Perplexity API key not found. Please set PERPLEXITY_API_KEY in your .env file.", "red")
        sys.exit(1)
    
    # Initialize the StockPredictor
    try:
        predictor = StockPredictor(
            model_type=args.model, 
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
    except ValueError as e:
        print_colored(f"Error: {str(e)}", "red")
        sys.exit(1)
    
    print_colored(f"Analyzing {ticker} using Perplexity {args.model} model...", "blue")
    
    # Get financial data
    print_colored(f"Fetching financial data for {ticker}...", "blue")
    try:
        financial_metrics = get_financial_metrics(ticker, period="ttm", limit=5)
        market_cap = get_market_cap(ticker)
    except Exception as e:
        error_msg = str(e)
        print_colored(f"Error fetching financial data: {error_msg}", "red")
        
        # Check for insufficient credits error
        if "Insufficient credits" in error_msg:
            print_colored("\nNOTE: The Financial Datasets API requires credits for most tickers.", "yellow")
            print_colored("Free tickers include: AAPL, GOOGL, MSFT, NVDA, and TSLA.", "yellow")
            print_colored("For other tickers, you need to purchase credits at https://financialdatasets.ai/", "yellow")
            
            # Ask if the user wants to continue with limited data
            continue_analysis = input("\nDo you want to continue analysis with limited data? (y/n): ").lower().strip() == 'y'
            if not continue_analysis:
                print_colored("Analysis cancelled.", "red")
                sys.exit(0)
        
        financial_metrics = []
        market_cap = None
    
    # Get price data
    print_colored(f"Fetching price data for {ticker}...", "blue")
    try:
        price_data = get_price_data(ticker, limit=30)
    except Exception as e:
        error_msg = str(e)
        print_colored(f"Error fetching price data: {error_msg}", "red")
        
        # Check for insufficient credits error
        if "Insufficient credits" in error_msg and not 'continue_analysis' in locals():
            print_colored("\nNOTE: The Financial Datasets API requires credits for most tickers.", "yellow")
            print_colored("Free tickers include: AAPL, GOOGL, MSFT, NVDA, and TSLA.", "yellow")
            print_colored("For other tickers, you need to purchase credits at https://financialdatasets.ai/", "yellow")
            
            # Ask if the user wants to continue with limited data
            continue_analysis = input("\nDo you want to continue analysis with limited data? (y/n): ").lower().strip() == 'y'
            if not continue_analysis:
                print_colored("Analysis cancelled.", "red")
                sys.exit(0)
        
        price_data = None
    
    # Prepare data for analysis
    data = {
        "financial_data": {
            "metrics": [metric.model_dump() for metric in financial_metrics] if financial_metrics else [],
            "market_cap": market_cap
        },
        "price_data": price_data.to_dict() if hasattr(price_data, 'to_dict') else {},
        "technical_indicators": {
            "moving_averages": {
                "sma_50": 150.25,  # Example values - in a real implementation, these would be calculated
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
    print_colored(f"Performing comprehensive analysis of {ticker}...", "blue")
    try:
        analysis = predictor.analyze_ticker(ticker, data)
    except Exception as e:
        print_colored(f"Error during analysis: {str(e)}", "red")
        sys.exit(1)
    
    # Print the results
    print_colored("\n=== ANALYSIS RESULTS ===\n", "green")
    
    # Print recommendation
    try:
        recommendation = json.loads(analysis["recommendation"])
        
        signal = recommendation['recommendation']['signal'].upper()
        confidence = recommendation['recommendation']['confidence']
        time_horizon = recommendation['recommendation']['time_horizon']
        
        # Color-code the signal
        if signal == "BULLISH":
            signal_color = "green"
        elif signal == "BEARISH":
            signal_color = "red"
        else:
            signal_color = "yellow"
        
        print_colored(f"RECOMMENDATION FOR {ticker}:", "cyan")
        print_colored(f"Signal: {signal}", signal_color)
        print_colored(f"Confidence: {confidence}%", "cyan")
        print_colored(f"Time Horizon: {time_horizon}", "cyan")
        
        print_colored("\nReasoning:", "cyan")
        print(recommendation['recommendation']['reasoning'])
        
        print_colored("\n=== SUMMARY ===", "green")
        print(recommendation['summary'])
        
        print_colored("\n=== RISKS ===", "yellow")
        print(recommendation['risks'])
        
        if args.verbose:
            print_colored("\n=== FUNDAMENTAL ANALYSIS ===", "green")
            fundamental = json.loads(analysis["fundamental_analysis"])
            print(json.dumps(fundamental, indent=2))
            
            print_colored("\n=== TECHNICAL ANALYSIS ===", "green")
            technical = json.loads(analysis["technical_analysis"])
            print(json.dumps(technical, indent=2))
            
            print_colored("\n=== SENTIMENT ANALYSIS ===", "green")
            sentiment = json.loads(analysis["sentiment_analysis"])
            print(json.dumps(sentiment, indent=2))
    
    except (json.JSONDecodeError, KeyError) as e:
        print_colored(f"Error parsing analysis results: {str(e)}", "red")
    
    # Save the full analysis to a file
    output_file = args.output or f"{ticker}_analysis.json"
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print_colored(f"\nFull analysis saved to {output_file}", "green")

if __name__ == "__main__":
    main()
