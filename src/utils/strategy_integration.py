"""Utilities for integrating trading strategies with the main application."""

from typing import Dict, Any
import json
from langchain_core.messages import HumanMessage

def register_strategies(state: Dict[str, Any]) -> Dict[str, Any]:
    """Register trading strategies in the initial state."""
    # Initialize strategies dictionary if it doesn't exist
    if "strategies" not in state["data"]:
        state["data"]["strategies"] = {}
    
    # Add any strategy-specific initialization here
    # For example, you could add configuration parameters for each strategy
    state["data"]["strategies"]["momentum"] = {
        "lookback_period": 180,  # Days to look back for momentum analysis
        "rsi_period": 14,        # Period for RSI calculation
        "macd_fast": 12,         # Fast period for MACD
        "macd_slow": 26,         # Slow period for MACD
        "macd_signal": 9,        # Signal period for MACD
        "atr_period": 14,        # Period for ATR calculation
    }
    
    state["data"]["strategies"]["value"] = {
        "margin_of_safety": 0.15,  # Required margin of safety for bullish signal
        "pe_threshold": 15,        # P/E ratio threshold for value consideration
        "pb_threshold": 1.5,       # P/B ratio threshold for value consideration
        "dividend_threshold": 3.0,  # Dividend yield threshold (%)
    }
    
    return state

def integrate_strategy_signals(state: Dict[str, Any]) -> Dict[str, Any]:
    """Integrate strategy signals with analyst signals for final decision making."""
    # Check if we have strategy signals
    if "analyst_signals" not in state["data"]:
        return state
    
    analyst_signals = state["data"]["analyst_signals"]
    
    # Extract momentum strategy signals if available
    momentum_signals = analyst_signals.get("momentum_strategy_agent", {})
    value_signals = analyst_signals.get("value_strategy_agent", {})
    
    # If we have both strategy signals, we can create a combined strategy signal
    if momentum_signals and value_signals:
        combined_signals = {}
        
        for ticker in momentum_signals.keys():
            if ticker in value_signals:
                momentum_signal = momentum_signals[ticker]
                value_signal = value_signals[ticker]
                
                # Create a combined signal based on both strategies
                combined_signal = combine_strategy_signals(ticker, momentum_signal, value_signal)
                combined_signals[ticker] = combined_signal
        
        # Add the combined signals to the analyst signals
        analyst_signals["combined_strategy"] = combined_signals
        
        # Add a message with the combined signals
        combined_message = HumanMessage(content=json.dumps(combined_signals), name="combined_strategy")
        state["messages"].append(combined_message)
    
    return state

def combine_strategy_signals(ticker: str, momentum_signal: Dict[str, Any], value_signal: Dict[str, Any]) -> Dict[str, Any]:
    """Combine momentum and value strategy signals for a ticker."""
    # Extract signals and confidence
    momentum_direction = momentum_signal.get("signal", "neutral")
    momentum_confidence = momentum_signal.get("confidence", 0)
    
    value_direction = value_signal.get("signal", "neutral")
    value_confidence = value_signal.get("confidence", 0)
    
    # Determine combined signal direction
    # If both agree, use that direction with higher confidence
    # If they disagree, use the one with higher confidence
    # If neutral, remain neutral
    
    if momentum_direction == value_direction:
        combined_direction = momentum_direction
        combined_confidence = max(momentum_confidence, value_confidence)
    elif momentum_direction == "neutral":
        combined_direction = value_direction
        combined_confidence = value_confidence * 0.8  # Reduce confidence slightly
    elif value_direction == "neutral":
        combined_direction = momentum_direction
        combined_confidence = momentum_confidence * 0.8  # Reduce confidence slightly
    else:
        # They disagree and neither is neutral
        if momentum_confidence > value_confidence:
            combined_direction = momentum_direction
            combined_confidence = momentum_confidence * 0.7  # Reduce confidence more
        else:
            combined_direction = value_direction
            combined_confidence = value_confidence * 0.7  # Reduce confidence more
    
    # Determine entry and exit prices
    # For entry price, use the higher of the two if bullish, lower if bearish
    if combined_direction == "bullish":
        entry_price = max(
            momentum_signal.get("entry_price", 0),
            value_signal.get("entry_price", 0)
        )
        # For exit price, use the higher of the two
        exit_price = max(
            momentum_signal.get("exit_price", 0),
            value_signal.get("exit_price", 0)
        )
        # For stop loss, use the higher of the two (more conservative)
        stop_loss = momentum_signal.get("stop_loss", 0)
    elif combined_direction == "bearish":
        entry_price = min(
            momentum_signal.get("entry_price", 0) or float('inf'),
            value_signal.get("entry_price", 0) or float('inf')
        )
        if entry_price == float('inf'):
            entry_price = 0
        # For exit price, use the lower of the two
        exit_price = min(
            momentum_signal.get("exit_price", 0) or float('inf'),
            value_signal.get("exit_price", 0) or float('inf')
        )
        if exit_price == float('inf'):
            exit_price = 0
        # For stop loss, use momentum stop loss
        stop_loss = momentum_signal.get("stop_loss", 0)
    else:
        # Neutral
        entry_price = momentum_signal.get("entry_price", 0)
        exit_price = momentum_signal.get("exit_price", 0)
        stop_loss = momentum_signal.get("stop_loss", 0)
    
    # Determine allocation based on value strategy
    allocation = value_signal.get("target_allocation", 0)
    
    # Create reasoning text
    if combined_direction == momentum_direction == value_direction:
        reasoning = f"Both momentum and value strategies agree on a {combined_direction} outlook. "
    elif momentum_direction == "neutral" and value_direction != "neutral":
        reasoning = f"Value strategy indicates {value_direction} while momentum is neutral. "
    elif value_direction == "neutral" and momentum_direction != "neutral":
        reasoning = f"Momentum strategy indicates {momentum_direction} while value assessment is neutral. "
    else:
        reasoning = f"Momentum strategy indicates {momentum_direction} while value strategy indicates {value_direction}. Using the higher confidence signal. "
    
    # Add details about the entry/exit strategy
    reasoning += f"Entry price set at ${entry_price:.2f} with target exit at ${exit_price:.2f}. "
    reasoning += f"Recommended allocation is {allocation:.1f}% of portfolio. "
    
    if stop_loss > 0:
        reasoning += f"Stop loss set at ${stop_loss:.2f} to limit downside risk."
    
    return {
        "signal": combined_direction,
        "confidence": combined_confidence,
        "reasoning": reasoning,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_loss": stop_loss,
        "target_allocation": allocation,
        "momentum_signal": momentum_direction,
        "momentum_confidence": momentum_confidence,
        "value_signal": value_direction,
        "value_confidence": value_confidence
    }
