# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import os
import json
import sys
from typing import List, Dict, Any

# Add the src directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import hedge fund components
from dotenv import load_dotenv
from main import run_hedge_fund
from utils.analysts import ANALYST_ORDER
from llm.models import LLM_ORDER, ModelProvider, get_model_info
from tools.api import get_prices, get_price_data, get_financial_metrics
from utils.stock_predictor import StockPredictor, PerplexityConfig

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Mashimi AI Hedge Fund",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .card {
        border-radius: 5px;
        padding: 20px;
        margin: 10px 0px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .bullish {
        color: #4CAF50;
        font-weight: bold;
    }
    .bearish {
        color: #F44336;
        font-weight: bold;
    }
    .neutral {
        color: #FFC107;
        font-weight: bold;
    }
    .info-text {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    # Temporarily revert to the remote logo until the local image issue is fixed
    st.image("https://raw.githubusercontent.com/mashimi/ai-hedge-fund/main/assets/logo.png", width=100)
    st.title("AI Hedge Fund")
    
    ticker_input = st.text_input("Enter Stock Ticker(s)", "AAPL", help="Enter one or more comma-separated tickers (e.g., AAPL,MSFT,NVDA)")
    
    # Date selection
    st.subheader("Analysis Period")
    end_date = st.date_input("End Date", datetime.now().date())
    start_date_options = ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
    lookback_period = st.selectbox("Lookback Period", start_date_options, index=2)
    
    if lookback_period == "1 Week":
        start_date = end_date - timedelta(days=7)
    elif lookback_period == "1 Month":
        start_date = end_date - relativedelta(months=1)
    elif lookback_period == "3 Months":
        start_date = end_date - relativedelta(months=3)
    elif lookback_period == "6 Months":
        start_date = end_date - relativedelta(months=6)
    else:  # 1 Year
        start_date = end_date - relativedelta(years=1)
    
    st.info(f"Analysis from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Select analysts
    st.subheader("Select Analysts")
    all_analysts = [value for _, value in ANALYST_ORDER]
    selected_analysts = []
    
    # Group analysts by category
    st.write("Investment Legends")
    buffett = st.checkbox("Warren Buffett", value=True)
    munger = st.checkbox("Charlie Munger", value=True)
    graham = st.checkbox("Ben Graham", value=True)
    ackman = st.checkbox("Bill Ackman", value=True)
    wood = st.checkbox("Cathie Wood", value=True)
    
    st.write("Analysis Styles")
    fundamentals = st.checkbox("Fundamentals", value=True)
    technicals = st.checkbox("Technicals", value=True)
    sentiment = st.checkbox("Sentiment", value=True)
    valuation = st.checkbox("Valuation", value=True)
    
    st.write("Trading Strategies")
    momentum_strategy = st.checkbox("Momentum Strategy", value=True, 
                                   help="Technical analysis strategy with entry/exit points and stop-loss levels")
    value_strategy = st.checkbox("Value Strategy", value=True,
                               help="Fundamental analysis strategy with portfolio allocation recommendations")
    
    # Add selected analysts to the list
    if buffett:
        selected_analysts.append("warren_buffett")
    if munger:
        selected_analysts.append("charlie_munger")
    if graham:
        selected_analysts.append("ben_graham")
    if ackman:
        selected_analysts.append("bill_ackman")
    if wood:
        selected_analysts.append("cathie_wood")
    if fundamentals:
        selected_analysts.append("fundamentals_analyst")
    if technicals:
        selected_analysts.append("technical_analyst")
    if sentiment:
        selected_analysts.append("sentiment_analyst")
    if valuation:
        selected_analysts.append("valuation_analyst")
    if momentum_strategy:
        selected_analysts.append("momentum_strategy")
    if value_strategy:
        selected_analysts.append("value_strategy")
    
    # LLM Selection
    st.subheader("Select LLM Model")
    model_options = [(display, model, provider) for display, model, provider in LLM_ORDER]
    default_model_index = next((i for i, (display, _, _) in enumerate(model_options) if "sonar-medium" in display), 0)
    selected_model_display = st.selectbox(
        "Model",
        options=[display for display, _, _ in model_options],
        index=default_model_index
    )
    
    # Extract model name and provider
    selected_model = next((model for display, model, _ in model_options if display == selected_model_display), None)
    selected_provider = next((provider for display, _, provider in model_options if display == selected_model_display), None)
    
    # StockPredictor options
    st.subheader("Advanced Analysis")
    use_stock_predictor = st.checkbox("Use StockPredictor (Mashimi Reasoning)", value=False, 
                                     help="Enable detailed stock analysis using Perplexity sonar-Reasoning")
    
    if use_stock_predictor:
        predictor_model_type = st.selectbox(
            "Predictor Model",
            options=list(PerplexityConfig.MODELS.keys()),
            index=0,
            help="Select the Perplexity sonar-reasoning model to use for StockPredictor analysis"
        )
        
        predictor_temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Temperature for sampling (higher = more creative, lower = more deterministic)"
        )
    
    # Run analysis button
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)
    
    # Disclaimer
    st.markdown("---")
    st.caption("**Disclaimer**: Trading is not for every one be responsible for your lost.")


# Helper functions
def format_signal(signal: str) -> str:
    """Format signal with appropriate CSS class."""
    if signal.lower() == "bullish":
        return f'<span class="bullish">BULLISH</span>'
    elif signal.lower() == "bearish":
        return f'<span class="bearish">BEARISH</span>'
    else:
        return f'<span class="neutral">NEUTRAL</span>'


def run_analysis(tickers: List[str], start_date: str, end_date: str, selected_analysts: List[str], model_name: str, model_provider: str) -> Dict[str, Any]:
    """Run the hedge fund analysis."""
    # Initialize portfolio with cash amount and stock positions
    portfolio = {
        "cash": 100000.0,  # Initial cash amount
        "margin_requirement": 0.0,  # Initial margin requirement
        "positions": {
            ticker: {
                "long": 0,  # Number of shares held long
                "short": 0,  # Number of shares held short
                "long_cost_basis": 0.0,  # Average cost basis for long positions
                "short_cost_basis": 0.0,  # Average price at which shares were sold short
            } for ticker in tickers
        },
        "realized_gains": {
            ticker: {
                "long": 0.0,  # Realized gains from long positions
                "short": 0.0,  # Realized gains from short positions
            } for ticker in tickers
        }
    }
    
    # Run the hedge fund
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=True,
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=model_provider,
    )
    
    return result


def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Get stock price data and return as DataFrame."""
    prices = get_price_data(ticker, start_date, end_date)
    return prices


def create_price_chart(df: pd.DataFrame, ticker: str) -> alt.Chart:
    """Create an interactive price chart with Altair."""
    source = df.reset_index()
    source = source.rename(columns={"index": "Date"})
    
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Date'], empty=False)
    
    # Base chart
    line = alt.Chart(source).mark_line().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('close:Q', title='Price', scale=alt.Scale(zero=False)),
        color=alt.value("#1E88E5")
    )
    
    # Points with tooltip
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    
    # Draw a rule at the location of the selection
    tooltips = alt.Chart(source).mark_rule().encode(
        x='Date:T',
        opacity=alt.condition(nearest, alt.value(0.5), alt.value(0)),
        tooltip=[
            alt.Tooltip('Date:T', title='Date'),
            alt.Tooltip('open:Q', title='Open', format='$.2f'),
            alt.Tooltip('high:Q', title='High', format='$.2f'),
            alt.Tooltip('low:Q', title='Low', format='$.2f'),
            alt.Tooltip('close:Q', title='Close', format='$.2f'),
            alt.Tooltip('volume:Q', title='Volume', format=',')
        ]
    ).add_selection(nearest)
    
    return (line + points + tooltips).properties(
        title=f"{ticker} Stock Price",
        width=800,
        height=400
    ).interactive()


def create_decision_card(ticker: str, decision: Dict[str, Any], analysis_data: Dict[str, Any]) -> None:
    """Create a card displaying the trading decision for a ticker."""
    action = decision.get("action", "").upper()
    quantity = decision.get("quantity", 0)
    confidence = decision.get("confidence", 0)
    reasoning = decision.get("reasoning", "")
    
    # Determine the action color
    if action == "BUY":
        action_color = "#4CAF50"  # Green
    elif action == "SELL":
        action_color = "#F44336"  # Red
    else:
        action_color = "#FFC107"  # Yellow
    
    st.markdown(f"""
    <div class="card" style="border-left: 5px solid {action_color};">
        <h3 style="color: {action_color};">{action} {quantity} shares</h3>
        <p>Confidence: {confidence:.1f}%</p>
        <p>{reasoning}</p>
    </div>
    """, unsafe_allow_html=True)


def create_analyst_signals_table(ticker: str, analyst_signals: Dict[str, Any]) -> None:
    """Create a table of analyst signals for a ticker."""
    data = []
    
    # Create separate lists for traditional analysts and strategies
    traditional_analysts = []
    strategy_signals = []
    
    for agent, signals in analyst_signals.items():
        if ticker in signals:
            signal_data = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            
            entry = {
                "Analyst": agent_name,
                "Signal": signal_data.get("signal", "").upper(),
                "Confidence": f"{signal_data.get('confidence', 0):.1f}%",
                "Details": signal_data.get("reasoning", "")
            }
            
            # Add to appropriate list based on agent type
            if agent in ["momentum_strategy_agent", "value_strategy_agent"]:
                strategy_signals.append(entry)
            else:
                traditional_analysts.append(entry)
    
    # Display traditional analysts
    if traditional_analysts:
        st.subheader("Analyst Signals")
        df_analysts = pd.DataFrame(traditional_analysts)
        st.dataframe(df_analysts, use_container_width=True)
    
    # Display strategy signals with additional information
    if strategy_signals:
        st.subheader("Strategy Signals")
        
        for strategy in strategy_signals:
            # Determine signal color
            signal = strategy["Signal"]
            if signal == "BULLISH":
                signal_color = "green"
            elif signal == "BEARISH":
                signal_color = "red"
            else:
                signal_color = "yellow"
            
            # Extract additional strategy-specific data
            agent_name = strategy["Analyst"]
            confidence = strategy["Confidence"]
            reasoning = strategy["Details"]
            
            # Get additional data based on strategy type
            additional_info = {}
            agent_key = agent_name.lower().replace(" ", "_") + "_agent"
            if ticker in analyst_signals.get(agent_key, {}):
                signal_data = analyst_signals[agent_key][ticker]
                
                if "momentum" in agent_name.lower():
                    additional_info = {
                        "Entry Price": f"${signal_data.get('entry_price', 0):.2f}",
                        "Exit Price": f"${signal_data.get('exit_price', 0):.2f}",
                        "Stop Loss": f"${signal_data.get('stop_loss', 0):.2f}"
                    }
                elif "value" in agent_name.lower():
                    additional_info = {
                        "Target Allocation": f"{signal_data.get('target_allocation', 0):.1f}%",
                        "Entry Price": f"${signal_data.get('entry_price', 0):.2f}",
                        "Exit Price": f"${signal_data.get('exit_price', 0):.2f}"
                    }
            
            # Create a card for the strategy
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid {signal_color};">
                <h4>{agent_name}</h4>
                <p>Signal: <span style="color: {signal_color}; font-weight: bold;">{signal}</span> (Confidence: {confidence})</p>
                <p><strong>Details:</strong> {reasoning}</p>
                <p><strong>Trading Parameters:</strong></p>
                <ul>
                    {"".join([f"<li><strong>{k}:</strong> {v}</li>" for k, v in additional_info.items()])}
                </ul>
            </div>
            """, unsafe_allow_html=True)


def run_stock_predictor_analysis(ticker: str, model_type: str, temperature: float) -> Dict[str, Any]:
    """Run the StockPredictor analysis for a ticker."""
    try:
        # Initialize the StockPredictor with increased timeout and retries
        st.info("Initializing StockPredictor...")
        predictor = StockPredictor(
            model_type=model_type, 
            temperature=temperature,
            timeout=120.0,  # 2 minutes timeout
            max_retries=5    # 5 retries
        )
        
        # Get financial data
        end_date_str = datetime.now().strftime("%Y-%m-%d")
        financial_metrics = get_financial_metrics(ticker, end_date_str, period="ttm", limit=5)
        market_cap = None
        try:
            market_cap = get_financial_metrics(ticker, end_date_str, "market_cap")[0].value
        except:
            pass
        
        # Get price data
        try:
            # Calculate start date (30 days before end date)
            start_date_str = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            price_data = get_price_data(ticker, start_date_str, end_date_str)
        except Exception as e:
            st.warning(f"Error fetching price data: {str(e)}")
            price_data = pd.DataFrame()  # Empty DataFrame
        
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
        analysis = predictor.analyze_ticker(ticker, data)
        return analysis
    except Exception as e:
        st.error(f"Error running StockPredictor analysis: {str(e)}")
        return {}


def display_stock_predictor_analysis(ticker: str, analysis: Dict[str, Any]) -> None:
    """Display the StockPredictor analysis results."""
    if not analysis:
        st.warning("No StockPredictor analysis available.")
        return
    
    # Check if all required keys are present
    required_keys = ["recommendation", "fundamental_analysis", "technical_analysis", "sentiment_analysis"]
    missing_keys = [key for key in required_keys if key not in analysis]
    if missing_keys:
        st.warning(f"Analysis is incomplete. Missing: {', '.join(missing_keys)}")
        return
    
    # Check if the recommendation contains an error message
    try:
        recommendation_data = json.loads(analysis["recommendation"])
        if "error" in recommendation_data:
            # Clean up HTML tags from error message
            import re
            error_msg = recommendation_data.get('message', 'Unknown error')
            clean_error = re.sub(r'<[^>]*>', '', error_msg)  # Remove HTML tags
            clean_error = re.sub(r'\s+', ' ', clean_error).strip()  # Remove extra whitespace
            
            st.error(f"Perplexity API Error: {clean_error}")
            st.info("""The analysis will continue with fallback data. You can:
1. Try again later when the API is available
2. Disable StockPredictor in the sidebar
3. Continue with other analysts' recommendations""")
        else:
            # Only show raw recommendation for debugging if there's no error
            st.write("Raw recommendation:")
            st.code(analysis["recommendation"])
    except (json.JSONDecodeError, KeyError):
        # If we can't parse the recommendation, just show it as is
        st.write("Raw recommendation:")
        st.code(analysis["recommendation"])
    
    try:
        # Parse the analysis results
        try:
            recommendation = json.loads(analysis["recommendation"])
        except (json.JSONDecodeError, TypeError) as e:
            st.error(f"Error parsing recommendation: {str(e)}")
            # Create a default recommendation
            recommendation = {
                "summary": "Analysis could not be completed due to insufficient data.",
                "agreements": "N/A",
                "conflicts": "N/A",
                "risks": "Unable to assess risks due to insufficient data.",
                "recommendation": {
                    "signal": "neutral",
                    "confidence": 50,
                    "reasoning": "Insufficient data to make a recommendation.",
                    "time_horizon": "medium term"
                }
            }
            
        try:
            fundamental = json.loads(analysis["fundamental_analysis"])
        except (json.JSONDecodeError, TypeError) as e:
            st.error(f"Error parsing fundamental analysis: {str(e)}")
            fundamental = {}
            
        try:
            technical = json.loads(analysis["technical_analysis"])
        except (json.JSONDecodeError, TypeError) as e:
            st.error(f"Error parsing technical analysis: {str(e)}")
            technical = {}
            
        try:
            sentiment = json.loads(analysis["sentiment_analysis"])
        except (json.JSONDecodeError, TypeError) as e:
            st.error(f"Error parsing sentiment analysis: {str(e)}")
            sentiment = {}
        
        # Display the recommendation
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
        
        st.markdown(f"""
        <div class="card">
            <h3> Reasoning Recommendation</h3>
            <p>Signal: <span style="color: {signal_color}; font-weight: bold;">{signal}</span></p>
            <p>Confidence: {confidence}%</p>
            <p>Time Horizon: {time_horizon}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display summary and risks
        st.markdown("### Summary")
        st.write(recommendation['summary'])
        
        st.markdown("### Risks")
        st.write(recommendation['risks'])
        
        # Create tabs for detailed analysis
        analysis_tabs = st.tabs(["Fundamental", "Technical", "Sentiment"])
        
        with analysis_tabs[0]:
            st.markdown("### Fundamental Analysis")
            st.json(fundamental)
        
        with analysis_tabs[1]:
            st.markdown("### Technical Analysis")
            st.json(technical)
        
        with analysis_tabs[2]:
            st.markdown("### Sentiment Analysis")
            st.json(sentiment)
    
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Error parsing StockPredictor analysis: {str(e)}")


def display_metrics(ticker: str, metrics: Dict[str, Any]) -> None:
    """Display key financial metrics for a ticker."""
    col1, col2, col3 = st.columns(3)
    
    if metrics:
        with col1:
            st.metric("P/E Ratio", f"{metrics.get('price_to_earnings_ratio', 'N/A'):.2f}" if metrics.get('price_to_earnings_ratio') is not None else "N/A")
            st.metric("ROE", f"{metrics.get('return_on_equity', 0) * 100:.2f}%" if metrics.get('return_on_equity') is not None else "N/A")
        
        with col2:
            st.metric("P/B Ratio", f"{metrics.get('price_to_book_ratio', 'N/A'):.2f}" if metrics.get('price_to_book_ratio') is not None else "N/A")
            st.metric("Debt to Equity", f"{metrics.get('debt_to_equity', 'N/A'):.2f}" if metrics.get('debt_to_equity') is not None else "N/A")
        
        with col3:
            st.metric("Operating Margin", f"{metrics.get('operating_margin', 0) * 100:.2f}%" if metrics.get('operating_margin') is not None else "N/A")
            st.metric("Revenue Growth", f"{metrics.get('revenue_growth', 0) * 100:.2f}%" if metrics.get('revenue_growth') is not None else "N/A")
            

# Main app
st.markdown('<h1 class="main-header">Mashimi AI Hedge Fund</h1>', unsafe_allow_html=True)

# Process tickers
if ticker_input:
    tickers = [ticker.strip() for ticker in ticker_input.split(",")]
    
    if run_button:
        st.markdown(f"<h2 class='sub-header'>Analysis Results</h2>", unsafe_allow_html=True)
        
        # Show spinner while running analysis
        with st.spinner("Running AI analysis..."):
            # Run the analysis
            result = run_analysis(
                tickers=tickers,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                selected_analysts=selected_analysts,
                model_name=selected_model,
                model_provider=selected_provider
            )
            
            decisions = result.get("decisions", {})
            analyst_signals = result.get("analyst_signals", {})
            
            # For each ticker, display a tab with analysis
            tabs = st.tabs(tickers)
            
            for i, ticker in enumerate(tickers):
                with tabs[i]:
                    # Create two columns for layout
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Display stock price chart
                        prices_df = get_stock_data(
                            ticker, 
                            start_date.strftime("%Y-%m-%d"), 
                            end_date.strftime("%Y-%m-%d")
                        )
                        
                        # Create and display the chart
                        chart = create_price_chart(prices_df, ticker)
                        st.altair_chart(chart, use_container_width=True)
                        
                        # Show analyst signals
                        st.markdown(f"<h3 class='sub-header'>Analyst Signals</h3>", unsafe_allow_html=True)
                        create_analyst_signals_table(ticker, analyst_signals)
                    
                    with col2:
                        # Display financial metrics
                        st.markdown(f"<h3 class='sub-header'>Financial Metrics</h3>", unsafe_allow_html=True)
                        
                        # Get financial metrics
                        financial_metrics = get_financial_metrics(ticker, end_date.strftime("%Y-%m-%d"))
                        if financial_metrics:
                            display_metrics(ticker, financial_metrics[0].model_dump())
                        
                        # Display the AI decision
                        st.markdown(f"<h3 class='sub-header'>Trading Decision</h3>", unsafe_allow_html=True)
                        
                        if ticker in decisions:
                            create_decision_card(ticker, decisions[ticker], analyst_signals)
                        else:
                            st.warning(f"No decision available for {ticker}")
                    
                    # Run StockPredictor analysis if enabled
                    if use_stock_predictor:
                        st.markdown(f"<h3 class='sub-header'>Mashimi Reasoning Analysis</h3>", unsafe_allow_html=True)
                        
                        with st.spinner(f"Running Perplexity {predictor_model_type} analysis for {ticker}..."):
                            predictor_analysis = run_stock_predictor_analysis(
                                ticker=ticker,
                                model_type=predictor_model_type,
                                temperature=predictor_temperature
                            )
                            
                            display_stock_predictor_analysis(ticker, predictor_analysis)
    else:
        # Display instructions when the app first loads
        st.info("Enter one or more stock tickers, select your analysis options, and click 'Run Analysis' to get started.")
        
        # Add some sample content to show what the app will do
        st.markdown("""
        ## Sample Analysis
        
        The Mashimi AI Hedge Fund  will:
        
        1. **Pull stock data** for your selected tickers
        2. **Analyze using multiple AI agents** with different investment philosophies:
           - Value investors (Buffett, Munger, Graham)
           - Growth investors (Wood)
           - Activist investors (Ackman)
           - Specialized analysts (Fundamentals, Technicals, Sentiment)
        3. **Generate trading recommendations** with confidence levels
        4. **Provide detailed rationale** for each decision
        
        ### Advanced Features
        
        Enable the **StockPredictor (Mashimi Reasoning)** option in the sidebar to get:
        
        - Comprehensive analysis using Mashimi Reasoning model
        - Detailed fundamental, technical, and sentiment analysis
        - Advanced recommendation with confidence level and time horizon
        - Summary of key points and risk assessment
        
        Click "Run Analysis" to see it in action!
        """)
