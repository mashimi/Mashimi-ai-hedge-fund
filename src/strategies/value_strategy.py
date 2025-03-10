from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from graph.state import AgentState, show_agent_reasoning
from pydantic import BaseModel, Field
from typing_extensions import Literal
from tools.api import get_financial_metrics, get_market_cap, search_line_items
import json
from utils.progress import progress
from utils.llm import call_llm


class ValueSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str
    target_allocation: float = Field(description="Suggested portfolio allocation percentage (0-100)")
    entry_price: float = Field(description="Suggested entry price")
    exit_price: float = Field(description="Suggested exit price")


def value_strategy_agent(state: AgentState):
    """Analyzes stocks using value investing principles and provides allocation recommendations."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Collect all analysis
    value_analysis = {}

    for ticker in tickers:
        progress.update_status("value_strategy_agent", ticker, "Fetching financial metrics")
        
        # Fetch financial metrics
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)
        
        progress.update_status("value_strategy_agent", ticker, "Gathering financial line items")
        
        # Fetch key financial line items - using a try/except block to handle API errors
        try:
            financial_line_items = search_line_items(
                ticker,
                [
                    "net_income",
                    "free_cash_flow",
                    "total_assets",
                    "total_liabilities",
                    "outstanding_shares",
                    "ebitda"
                ],
                end_date,
                period="ttm",
                limit=5,
            )
        except Exception as e:
            progress.update_status("value_strategy_agent", ticker, f"Error: {str(e)}")
            # Create a default response with empty data
            value_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 50.0,
                "reasoning": f"Unable to fetch financial data: {str(e)}",
                "target_allocation": 0.0,
                "entry_price": 0.0,
                "exit_price": 0.0
            }
            continue
        
        progress.update_status("value_strategy_agent", ticker, "Getting market cap")
        
        # Get current market cap
        market_cap = get_market_cap(ticker, end_date)
        
        progress.update_status("value_strategy_agent", ticker, "Calculating valuation metrics")
        
        # Calculate valuation metrics
        valuation_data = calculate_valuation_metrics(metrics, financial_line_items, market_cap)
        
        progress.update_status("value_strategy_agent", ticker, "Generating value signal")
        
        # Generate value signal
        signal = generate_value_signal(
            ticker=ticker,
            valuation_data=valuation_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        # Store analysis
        value_analysis[ticker] = {
            "signal": signal.signal,
            "confidence": signal.confidence,
            "reasoning": signal.reasoning,
            "target_allocation": signal.target_allocation,
            "entry_price": signal.entry_price,
            "exit_price": signal.exit_price
        }
        
        progress.update_status("value_strategy_agent", ticker, "Done")

    # Create the message
    message = HumanMessage(content=json.dumps(value_analysis), name="value_strategy_agent")

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(value_analysis, "Value Strategy Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["value_strategy_agent"] = value_analysis

    return {
        "messages": state["messages"] + [message],
        "data": state["data"],
    }


def calculate_valuation_metrics(metrics, financial_line_items, market_cap):
    """Calculate various valuation metrics from financial data."""
    
    # Initialize result dictionary
    result = {
        "market_cap": market_cap,
        "valuation_ratios": {},
        "historical_metrics": {},
        "intrinsic_value": None,
        "margin_of_safety": None,
        "value_score": 0
    }
    
    # Check if we have sufficient data
    if not metrics or not financial_line_items or not market_cap:
        return result
    
    # Get latest metrics and financial data
    latest_metrics = metrics[0] if metrics else None
    latest_financials = financial_line_items[0] if financial_line_items else None
    
    # Calculate valuation ratios
    if latest_metrics and latest_financials:
        # P/E Ratio
        if hasattr(latest_financials, 'net_income') and latest_financials.net_income and hasattr(latest_financials, 'outstanding_shares') and latest_financials.outstanding_shares:
            eps = latest_financials.net_income / latest_financials.outstanding_shares
            pe_ratio = market_cap / (latest_financials.net_income) if latest_financials.net_income > 0 else None
            result["valuation_ratios"]["pe_ratio"] = pe_ratio
            
            # Add to value score if P/E is attractive (below 15)
            if pe_ratio and pe_ratio < 15:
                result["value_score"] += 2
            elif pe_ratio and pe_ratio < 20:
                result["value_score"] += 1
        
        # P/B Ratio - using metrics instead since we don't have total_equity in line items
        if hasattr(latest_metrics, 'price_to_book_ratio') and latest_metrics.price_to_book_ratio:
            pb_ratio = latest_metrics.price_to_book_ratio
            result["valuation_ratios"]["pb_ratio"] = pb_ratio
            
            # Add to value score if P/B is attractive (below 1.5)
            if pb_ratio and pb_ratio < 1.5:
                result["value_score"] += 2
            elif pb_ratio and pb_ratio < 3:
                result["value_score"] += 1
        
        # EV/EBITDA
        if hasattr(latest_financials, 'ebitda') and latest_financials.ebitda:
            # Enterprise Value = Market Cap + Total Debt - Cash
            total_debt = latest_financials.total_liabilities if hasattr(latest_financials, 'total_liabilities') and latest_financials.total_liabilities else 0
            cash = latest_financials.total_assets - latest_financials.total_liabilities if hasattr(latest_financials, 'total_assets') and latest_financials.total_assets and hasattr(latest_financials, 'total_liabilities') and latest_financials.total_liabilities else 0
            enterprise_value = market_cap + total_debt - cash
            
            ev_ebitda = enterprise_value / latest_financials.ebitda if latest_financials.ebitda > 0 else None
            result["valuation_ratios"]["ev_ebitda"] = ev_ebitda
            
            # Add to value score if EV/EBITDA is attractive (below 8)
            if ev_ebitda and ev_ebitda < 8:
                result["value_score"] += 2
            elif ev_ebitda and ev_ebitda < 12:
                result["value_score"] += 1
        
        # Dividend Yield - using metrics instead
        if hasattr(latest_metrics, 'dividend_yield') and latest_metrics.dividend_yield:
            dividend_yield = latest_metrics.dividend_yield * 100  # Convert to percentage
            result["valuation_ratios"]["dividend_yield"] = dividend_yield
            
            # Add to value score if dividend yield is attractive (above 3%)
            if dividend_yield > 3:
                result["value_score"] += 2
            elif dividend_yield > 1.5:
                result["value_score"] += 1
    
    # Calculate historical growth rates if we have enough data
    if len(financial_line_items) >= 3:
        # Earnings Growth - only use net_income since we know it exists
        earnings_values = [item.net_income for item in financial_line_items if hasattr(item, 'net_income') and item.net_income]
        if len(earnings_values) >= 3:
            earnings_growth = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1]) if earnings_values[-1] != 0 else 0
            result["historical_metrics"]["earnings_growth"] = earnings_growth
            
            # Add to value score if earnings growth is strong (above 15%)
            if earnings_growth > 0.15:
                result["value_score"] += 2
            elif earnings_growth > 0.08:
                result["value_score"] += 1
    
    # Calculate intrinsic value using Discounted Cash Flow (DCF)
    if (latest_financials and 
        hasattr(latest_financials, 'free_cash_flow') and latest_financials.free_cash_flow and 
        hasattr(latest_financials, 'outstanding_shares') and latest_financials.outstanding_shares):
        
        # Simple DCF calculation
        fcf_per_share = latest_financials.free_cash_flow / latest_financials.outstanding_shares
        growth_rate = 0.05  # Assume 5% growth for 10 years, then 3% terminal
        discount_rate = 0.09  # 9% discount rate
        
        # Calculate present value of future cash flows (10 years)
        present_value = 0
        for year in range(1, 11):
            future_fcf = fcf_per_share * (1 + growth_rate) ** year
            present_value += future_fcf / (1 + discount_rate) ** year
        
        # Terminal value (Gordon Growth Model)
        terminal_growth = 0.03  # 3% long-term growth
        terminal_value = (fcf_per_share * (1 + growth_rate) ** 10 * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        terminal_value_discounted = terminal_value / (1 + discount_rate) ** 10
        
        # Intrinsic value per share
        intrinsic_value_per_share = present_value + terminal_value_discounted
        
        # Total intrinsic value
        intrinsic_value = intrinsic_value_per_share * latest_financials.outstanding_shares
        result["intrinsic_value"] = intrinsic_value
        
        # Calculate margin of safety
        current_share_price = market_cap / latest_financials.outstanding_shares
        margin_of_safety = (intrinsic_value_per_share - current_share_price) / current_share_price if current_share_price > 0 else 0
        result["margin_of_safety"] = margin_of_safety
        
        # Add to value score if margin of safety is high (above 30%)
        if margin_of_safety > 0.3:
            result["value_score"] += 3
        elif margin_of_safety > 0.15:
            result["value_score"] += 2
        elif margin_of_safety > 0:
            result["value_score"] += 1
    
    # Return the calculated metrics
    return result


def generate_value_signal(
    ticker: str,
    valuation_data: dict,
    model_name: str,
    model_provider: str,
) -> ValueSignal:
    """Generate value investing signal using LLM reasoning."""
    
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a value investing strategy AI agent. Your task is to analyze valuation metrics and generate precise investment signals with allocation recommendations.

            Follow these value investing principles:
            1. Margin of Safety: Buy significantly below intrinsic value
            2. Quality Business: Focus on companies with strong fundamentals
            3. Reasonable Valuation: Look for attractive P/E, P/B, and EV/EBITDA ratios
            4. Sustainable Competitive Advantage: Prefer companies with economic moats
            5. Long-term Perspective: Invest for the long term, not short-term gains
            6. Dividend Income: Consider dividend yield as part of total return
            7. Asset Allocation: Recommend higher allocations to more undervalued stocks

            Rules:
            - Bullish signal when value_score > 8 and margin_of_safety > 0.15
            - Bearish signal when value_score < 4 or margin_of_safety < -0.1
            - Neutral signal when conditions are mixed or inconclusive
            - Target allocation should be higher for more undervalued stocks
            - Entry price should be current price or lower
            - Exit price should be based on intrinsic value or valuation targets
            """
        ),
        (
            "human",
            """Based on the following valuation data for {ticker}, generate a value investing signal with allocation recommendation.

            Valuation Data:
            {valuation_data}

            Return the value signal in the following JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string",
              "target_allocation": float (0-100),
              "entry_price": float,
              "exit_price": float
            }}
            """
        ),
    ])

    # Generate the prompt
    prompt = template.invoke({
        "valuation_data": json.dumps(valuation_data, indent=2),
        "ticker": ticker
    })

    # Create default factory for ValueSignal
    def create_default_value_signal():
        # Calculate a default entry price (current market price)
        market_cap = valuation_data.get("market_cap", 0)
        outstanding_shares = 0
        for item in valuation_data.get("financial_line_items", [{}]):
            if item.get("outstanding_shares"):
                outstanding_shares = item.get("outstanding_shares")
                break
        
        current_price = market_cap / outstanding_shares if outstanding_shares > 0 else 100.0
        
        return ValueSignal(
            signal="neutral", 
            confidence=0.0, 
            reasoning="Error in analysis, defaulting to neutral",
            target_allocation=0.0,
            entry_price=current_price,
            exit_price=current_price * 1.2
        )

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=ValueSignal,
        agent_name="value_strategy_agent",
        default_factory=create_default_value_signal,
    )
