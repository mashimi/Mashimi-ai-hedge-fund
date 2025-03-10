from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from tools.api import get_prices, prices_to_df
import json
import pandas as pd
import numpy as np
from utils.progress import progress
from utils.llm import call_llm


class MomentumSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str
    entry_price: float = Field(description="Suggested entry price")
    exit_price: float = Field(description="Suggested exit price")
    stop_loss: float = Field(description="Suggested stop loss price")


def momentum_strategy_agent(state: AgentState):
    """Analyzes stocks using momentum indicators and provides entry/exit points."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Collect all analysis
    momentum_analysis = {}

    for ticker in tickers:
        progress.update_status("momentum_strategy_agent", ticker, "Fetching price data")
        
        # Get price data for the last 180 days
        try:
            prices = get_prices(
                ticker=ticker,
                start_date=data["start_date"],
                end_date=end_date,
            )
        except Exception as e:
            print(f"Price data error for {ticker}: {str(e)}")
            prices = []

        if not prices:
            progress.update_status("momentum_strategy_agent", ticker, "Failed: No price data found")
            # Create a default response with empty data
            momentum_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 20.0,
                "reasoning": "Unable to generate a reliable trading signal due to insufficient price data for momentum analysis. Without access to key momentum indicators like RSI, MACD, and moving averages, it would be speculative to provide directional bias. Recommending neutral stance until more data becomes available.",
                "entry_price": 0.0,
                "exit_price": 0.0,
                "stop_loss": 0.0
            }
            continue

        prices_df = prices_to_df(prices)
        
        progress.update_status("momentum_strategy_agent", ticker, "Calculating momentum indicators")
        
        # Calculate momentum indicators
        momentum_data = calculate_momentum_indicators(prices_df)
        
        progress.update_status("momentum_strategy_agent", ticker, "Generating momentum signal")
        
        # Generate momentum signal
        signal = generate_momentum_signal(
            ticker=ticker,
            momentum_data=momentum_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        # Store analysis
        momentum_analysis[ticker] = {
            "signal": signal.signal,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
            "entry_price": signal.entry_price,
            "exit_price": signal.exit_price,
            "stop_loss": signal.stop_loss
        }
        
        progress.update_status("momentum_strategy_agent", ticker, "Done")

    # Create the message
    message = HumanMessage(content=json.dumps(momentum_analysis), name="momentum_strategy_agent")

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(momentum_analysis, "Momentum Strategy Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["momentum_strategy_agent"] = momentum_analysis

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def calculate_momentum_indicators(prices_df: pd.DataFrame) -> dict:
    """Calculate various momentum indicators from price data."""
    df = prices_df.copy()
    
    # Ensure we have enough data
    if len(df) < 50:
        return {"error": "Insufficient price data for momentum analysis"}
    
    # Calculate RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD (Moving Average Convergence Divergence)
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['signal_line']
    
    # Calculate Moving Averages
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()
    
    # Calculate Rate of Change (ROC)
    df['roc'] = df['close'].pct_change(periods=10) * 100
    
    # Calculate Average Directional Index (ADX)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Calculate Bollinger Bands
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['stddev'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['sma20'] + (df['stddev'] * 2)
    df['lower_band'] = df['sma20'] - (df['stddev'] * 2)
    
    # Calculate momentum score
    df['momentum_score'] = 0
    
    # RSI component (30-70 range)
    df.loc[df['rsi'] < 30, 'momentum_score'] += 1  # Oversold
    df.loc[df['rsi'] > 70, 'momentum_score'] -= 1  # Overbought
    
    # MACD component
    df.loc[df['macd'] > df['signal_line'], 'momentum_score'] += 1  # Bullish
    df.loc[df['macd'] < df['signal_line'], 'momentum_score'] -= 1  # Bearish
    
    # Moving Average component
    df.loc[df['close'] > df['sma50'], 'momentum_score'] += 1  # Above 50-day MA
    df.loc[df['close'] < df['sma50'], 'momentum_score'] -= 1  # Below 50-day MA
    
    df.loc[df['sma20'] > df['sma50'], 'momentum_score'] += 1  # Golden cross
    df.loc[df['sma20'] < df['sma50'], 'momentum_score'] -= 1  # Death cross
    
    # ROC component
    df.loc[df['roc'] > 0, 'momentum_score'] += 1  # Positive momentum
    df.loc[df['roc'] < 0, 'momentum_score'] -= 1  # Negative momentum
    
    # Get the latest values
    latest = df.iloc[-1].to_dict()
    
    # Calculate suggested entry, exit, and stop loss prices
    current_price = latest['close']
    atr = latest.get('atr', current_price * 0.02)  # Default to 2% if ATR not available
    
    # Entry price is the current price
    entry_price = current_price
    
    # Exit price based on momentum score and ATR
    if latest['momentum_score'] > 0:
        # Bullish: exit at 3 ATR above entry
        exit_price = entry_price + (3 * atr)
        # Stop loss at 1.5 ATR below entry
        stop_loss = entry_price - (1.5 * atr)
    elif latest['momentum_score'] < 0:
        # Bearish: exit at 3 ATR below entry (for short positions)
        exit_price = entry_price - (3 * atr)
        # Stop loss at 1.5 ATR above entry (for short positions)
        stop_loss = entry_price + (1.5 * atr)
    else:
        # Neutral: symmetric exit points
        exit_price = entry_price * 1.05  # 5% profit target
        stop_loss = entry_price * 0.97   # 3% stop loss
    
    # Prepare the result
    result = {
        'current_price': current_price,
        'rsi': latest['rsi'],
        'macd': latest['macd'],
        'signal_line': latest['signal_line'],
        'macd_histogram': latest['macd_histogram'],
        'sma20': latest['sma20'],
        'sma50': latest['sma50'],
        'sma200': latest.get('sma200', None),
        'roc': latest['roc'],
        'atr': atr,
        'upper_band': latest['upper_band'],
        'lower_band': latest['lower_band'],
        'momentum_score': latest['momentum_score'],
        'entry_price': entry_price,
        'exit_price': exit_price,
        'stop_loss': stop_loss,
        'historical_data': {
            'close': df['close'].tolist()[-30:],  # Last 30 days of closing prices
            'rsi': df['rsi'].tolist()[-30:],      # Last 30 days of RSI
            'macd': df['macd'].tolist()[-30:],    # Last 30 days of MACD
        }
    }
    
    return result


def generate_momentum_signal(
    ticker: str,
    momentum_data: dict,
    model_name: str,
    model_provider: str,
) -> MomentumSignal:
    """Generate momentum trading signal using LLM reasoning."""
    
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a momentum trading strategy AI agent. Your task is to analyze momentum indicators and generate precise trading signals with entry and exit points.

            Follow these momentum trading principles:
            1. Trend Following: Identify and follow established trends
            2. Breakout Detection: Spot price breakouts from consolidation patterns
            3. Overbought/Oversold Conditions: Use RSI to identify extremes
            4. Moving Average Crossovers: Analyze golden and death crosses
            5. MACD Signals: Look for MACD line crossing signal line
            6. Risk Management: Set clear stop-loss levels based on ATR
            7. Profit Targets: Define realistic exit points based on momentum

            Rules:
            - Bullish signal when momentum_score > 2, RSI < 70, and price > SMA50
            - Bearish signal when momentum_score < -2, RSI > 30, and price < SMA50
            - Neutral signal when conditions are mixed or inconclusive
            - Entry price should be current price or next significant level
            - Exit price should be based on risk:reward ratio of at least 2:1
            - Stop loss should be based on ATR or recent swing points
            """
        ),
        (
            "human",
            """Based on the following momentum data for {ticker}, generate a trading signal with entry and exit points.

            Momentum Data:
            {momentum_data}

            Return the trading signal in the following JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string",
              "entry_price": float,
              "exit_price": float,
              "stop_loss": float
            }}
            """
        ),
    ])

    # Generate the prompt
    prompt = template.invoke({
        "momentum_data": json.dumps(momentum_data, indent=2),
        "ticker": ticker
    })

    # Create default factory for MomentumSignal
    def create_default_momentum_signal():
        current_price = momentum_data.get('current_price', 100.0)
        return MomentumSignal(
            signal="neutral", 
            confidence=0.0, 
            reasoning="Error in analysis, defaulting to neutral",
            entry_price=current_price,
            exit_price=current_price * 1.05,
            stop_loss=current_price * 0.95
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=MomentumSignal,
        agent_name="momentum_strategy_agent",
        default_factory=create_default_momentum_signal,
    )
