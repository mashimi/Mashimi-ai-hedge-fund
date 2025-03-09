Complete AI Hedge Fund Codebase with Dashboard and Perplexity Integration
I've prepared the complete codebase for you to implement. Each section contains the full content of a file along with its path in the project structure. This allows you to simply copy each file to the appropriate location.
Project Structure
Copyai-hedge-fund/
├── dashboard.py
├── .env.example
├── pyproject.toml
├── README.md
├── LICENSE
└── src/
    ├── backtester.py
    ├── main.py
    ├── agents/
    │   ├── ben_graham.py
    │   ├── bill_ackman.py
    │   ├── cathie_wood.py
    │   ├── charlie_munger.py
    │   ├── fundamentals.py
    │   ├── portfolio_manager.py
    │   ├── risk_manager.py
    │   ├── sentiment.py
    │   ├── technicals.py
    │   ├── valuation.py
    │   └── warren_buffett.py
    ├── data/
    │   ├── cache.py
    │   └── models.py
    ├── graph/
    │   └── state.py
    ├── llm/
    │   ├── models.py
    │   ├── perplexity.py
    │   └── perplexity_langchain.py
    ├── tools/
    │   └── api.py
    └── utils/
        ├── __init__.py
        ├── analysts.py
        ├── display.py
        ├── llm.py
        ├── progress.py
        └── visualize.py
New Files
1. Dashboard.py (Root directory)
pythonCopy# dashboard.py
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

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Hedge Fund Dashboard",
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
    st.image("https://raw.githubusercontent.com/virattt/ai-hedge-fund/main/assets/logo.png", width=100)
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
    
    # Run analysis button
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)
    
    # Disclaimer
    st.markdown("---")
    st.caption("**Disclaimer**: This tool is for educational purposes only. Not financial advice.")


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
    
    for agent, signals in analyst_signals.items():
        if ticker in signals:
            signal_data = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            
            data.append({
                "Analyst": agent_name,
                "Signal": signal_data.get("signal", "").upper(),
                "Confidence": f"{signal_data.get('confidence', 0):.1f}%",
                "Details": signal_data.get("reasoning", "")
            })
    
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)


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
st.markdown('<h1 class="main-header">Mashimi AI Hedge Fund </h1>', unsafe_allow_html=True)

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
    else:
        # Display instructions when the app first loads
        st.info("Enter one or more stock tickers, select your analysis options, and click 'Run Analysis' to get started.")
        
        # Add some sample content to show what the app will do
        st.markdown("""
        ## Sample Analysis
        
        The AI Hedge Fund Dashboard will:
        
        1. **Pull stock data** for your selected tickers
        2. **Analyze using multiple AI agents** with different investment philosophies:
           - Value investors (Buffett, Munger, Graham)
           - Growth investors (Wood)
           - Activist investors (Ackman)
           - Specialized analysts (Fundamentals, Technicals, Sentiment)
        3. **Generate trading recommendations** with confidence levels
        4. **Provide detailed rationale** for each decision
        
        Click "Run Analysis" to see it in action!
        """)
2. src/llm/perplexity.py
pythonCopy"""
Perplexity Sonar Reasoning model client for the AI Hedge Fund.
"""
import os
import requests
import json
from typing import Dict, Any, Optional


class PerplexityClient:
    """Client for accessing Perplexity Sonar Reasoning model."""
    
    API_URL = "https://api.perplexity.ai/chat/completions"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Perplexity client with API key."""
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("Perplexity API key not found. Please set PERPLEXITY_API_KEY in your .env file.")
    
    def completion(self, messages: list[Dict[str, Any]], model: str = "sonar-medium-online") -> Dict[str, Any]:
        """
        Call the Perplexity API with messages.
        
        Args:
            messages: List of message dictionaries (role + content)
            model: Perplexity model to use (default: sonar-medium-online)
            
        Returns:
            API response as dictionary
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        response = requests.post(
            self.API_URL,
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Error from Perplexity API: {response.status_code} - {response.text}")
        
        return response.json()
3. src/llm/perplexity_langchain.py
pythonCopy"""
LangChain integration for Perplexity Sonar Reasoning.
"""
from typing import Any, Dict, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from llm.perplexity import PerplexityClient


class ChatPerplexity(BaseChatModel):
    """LangChain integration for the Perplexity Sonar Reasoning models."""
    
    client: PerplexityClient
    model_name: str = "sonar-medium-online"
    
    def __init__(
        self, 
        model: str = "sonar-medium-online",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize with model and API key."""
        super().__init__(**kwargs)
        self.client = PerplexityClient(api_key=api_key)
        self.model_name = model
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "perplexity-sonar"
    
    def _convert_messages_to_perplexity_format(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Convert LangChain messages to Perplexity format."""
        perplexity_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                perplexity_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                perplexity_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                perplexity_messages.append({"role": "system", "content": message.content})
            else:
                raise ValueError(f"Message type {type(message)} not supported")
        return perplexity_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response from Perplexity Sonar."""
        perplexity_messages = self._convert_messages_to_perplexity_format(messages)
        
        # Call the Perplexity API
        response = self.client.completion(
            messages=perplexity_messages,
            model=self.model_name
        )
        
        # Extract the response text
        response_text = response["choices"][0]["message"]["content"]
        
        # Create a LangChain chat generation object
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
Modified Files
1. src/llm/models.py
pythonCopyimport os
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from enum import Enum
from pydantic import BaseModel
from typing import Tuple
from llm.perplexity_langchain import ChatPerplexity


class ModelProvider(str, Enum):
    """Enum for supported LLM providers"""
    OPENAI = "OpenAI"
    GROQ = "Groq"
    ANTHROPIC = "Anthropic"
    PERPLEXITY = "Perplexity"  # Added Perplexity as a provider


class LLMModel(BaseModel):
    """Represents an LLM model configuration"""
    display_name: str
    model_name: str
    provider: ModelProvider

    def to_choice_tuple(self) -> Tuple[str, str, str]:
        """Convert to format needed for questionary choices"""
        return (self.display_name, self.model_name, self.provider.value)
    
    def is_deepseek(self) -> bool:
        """Check if the model is a DeepSeek model"""
        return self.model_name.startswith("deepseek")

    def is_perplexity(self) -> bool:
        """Check if the model is a Perplexity model"""
        return self.provider == ModelProvider.PERPLEXITY


# Define available models including new Perplexity Sonar models
AVAILABLE_MODELS = [
    LLMModel(
        display_name="[anthropic] claude-3.5-haiku",
        model_name="claude-3-5-haiku-latest",
        provider=ModelProvider.ANTHROPIC
    ),
    LLMModel(
        display_name="[anthropic] claude-3.5-sonnet",
        model_name="claude-3-5-sonnet-latest",
        provider=ModelProvider.ANTHROPIC
    ),
    LLMModel(
        display_name="[anthropic] claude-3.7-sonnet",
        model_name="claude-3-7-sonnet-latest",
        provider=ModelProvider.ANTHROPIC
    ),
    LLMModel(
        display_name="[groq] deepseek-r1 70b",
        model_name="deepseek-r1-distill-llama-70b",
        provider=ModelProvider.GROQ
    ),
    LLMModel(
        display_name="[groq] llama-3.3 70b",
        model_name="llama-3.3-70b-versatile",
        provider=ModelProvider.GROQ
    ),
    LLMModel(
        display_name="[openai] gpt-4o",
        model_name="gpt-4o",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[openai] gpt-4o-mini",
        model_name="gpt-4o-mini",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[openai] o1",
        model_name="o1",
        provider=ModelProvider.OPENAI
    ),
    LLMModel(
        display_name="[openai] o3-mini",
        model_name="o3-mini",
        provider=ModelProvider.OPENAI
    ),
    # New Perplexity Sonar models
    LLMModel(
        display_name="[perplexity] sonar-small-online",
        model_name="sonar-small-online",
        provider=ModelProvider.PERPLEXITY
    ),
    LLMModel(
        display_name="[perplexity] sonar-medium-online",
        model_name="sonar-medium-online",
        provider=ModelProvider.PERPLEXITY
    ),
    LLMModel(
        display_name="[perplexity] sonar-large-online",
        model_name="sonar-large-online",
        provider=ModelProvider.PERPLEXITY
    ),
]

# Create LLM_ORDER in the format expected by the UI
LLM_ORDER = [model.to_choice_tuple() for model in AVAILABLE_MODELS]

def get_model_info(model_name: str) -> LLMModel | None:
    """Get model information by model_name"""
    return next((model for model in AVAILABLE_MODELS if model.model_name == model_name), None)

def get_model(model_name: str, model_provider: ModelProvider) -> ChatOpenAI | ChatGroq | ChatAnthropic | ChatPerplexity | None:
    """Get the appropriate LLM model based on provider and model name"""
    if model_provider == ModelProvider.GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure GROQ_API_KEY is set in your .env file.")
            raise ValueError("Groq API key not found. Please make sure GROQ_API_KEY is set in your .env file.")
        return ChatGroq(model=model_name, api_key=api_key)
    
    elif model_provider == ModelProvider.OPENAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure OPENAI_API_KEY is set in your .env file.")
            raise ValueError("OpenAI API key not found. Please make sure OPENAI_API_KEY is set in your .env file.")
        return ChatOpenAI(model=model_name, api_key=api_key)
    
    elif model_provider == ModelProvider.ANTHROPIC:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure ANTHROPIC_API_KEY is set in your .env file.")
            raise ValueError("Anthropic API key not found. Please make sure ANTHROPIC_API_KEY is set in your .env file.")
        return ChatAnthropic(model=model_name, api_key=api_key)
    
    elif model_provider == ModelProvider.PERPLEXITY:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            print(f"API Key Error: Please make sure PERPLEXITY_API_KEY is set in your .env file.")
            raise ValueError("Perplexity API key not found. Please make sure PERPLEXITY_API_KEY is set in your .env file.")
        return ChatPerplexity(model=model_name, api_key=api_key)
    
    return None
2. src/utils/llm.py
pythonCopy"""Helper functions for LLM"""

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from utils.progress import progress

T = TypeVar('T', bound=BaseModel)

def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
    """
    Makes an LLM call with retry logic, handling both Deepseek and Perplexity models.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        model_provider: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure
        
    Returns:
        An instance of the specified Pydantic model
    """
    from llm.models import get_model, get_model_info
    
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)
    
    # For standard models (non-Deepseek, non-Perplexity), we can use structured output
    if model_info and not (model_info.is_deepseek() or model_info.is_perplexity()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )
    
    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)
            
            # For Deepseek, we need to extract and parse the JSON manually
            if model_info and model_info.is_deepseek():
                parsed_result = extract_json_from_deepseek_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            # For Perplexity, we also need to extract and parse JSON manually
            elif model_info and model_info.is_perplexity():
                parsed_result = extract_json_from_perplexity_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
            else:
                return result
                
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
            
            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)

def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None
    
    return model_class(**default_values)

def extract_json_from_deepseek_response(content: str) -> Optional[dict]:
    """Extracts JSON from Deepseek's markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from Deepseek response: {e}")
    return None

def extract_json_from_perplexity_response(content: str) -> Optional[dict]:
    """Extracts JSON from Perplexity's response, which could be markdown-formatted or direct JSON."""
    try:
        # First try to parse as direct JSON
        try:
            parsed_json = json.loads(content)
            return parsed_json
        except json.JSONDecodeError:
            pass
        
        # Then try to extract JSON from markdown code blocks
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
        
        # Finally try to extract JSON from any curly braces
        first_brace = content.find("{")
        last_brace = content.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_text = content[first_brace:last_brace + 1]
            return json.loads(json_text)
            
    except Exception as e:
        print(f"Error extracting JSON from Perplexity response: {e}")
    
    return None
3. src/agents/warren_buffett.py (Enhanced version)
pythonCopyfrom graph.state import AgentState, show_agent_reasoning
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from tools.api import get_financial_metrics, get_market_cap, search_line_items
from utils.llm import call_llm
from utils.progress import progress


class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def warren_buffett_agent(state: AgentState):
    """Analyzes stocks using Buffett's principles and LLM reasoning."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Collect all analysis for LLM reasoning
    analysis_data = {}
    buffett_analysis = {}

    for ticker in tickers:
        progress.update_status("warren_buffett_agent", ticker, "Fetching financial metrics")
        # Fetch required data
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)

        progress.update_status("warren_buffett_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization",
                "net_income",
                "outstanding_shares",
                "total_assets",
                "total_liabilities",
            ],
            end_date,
            period="ttm",
            limit=5,
        )

        progress.update_status("warren_buffett_agent", ticker, "Getting market cap")
        # Get current market cap
        market_cap = get_market_cap(ticker, end_date)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing fundamentals")
        # Analyze fundamentals
        fundamental_analysis = analyze_fundamentals(metrics)

        progress.update_status("warren_buffett_agent", ticker, "Analyzing consistency")
        consistency_analysis = analyze_consistency(financial_line_items)

        progress.update_status("warren_buffett_agent", ticker, "Calculating intrinsic value")
        intrinsic_value_analysis = calculate_intrinsic_value(financial_line_items)

        # Calculate total score
        total_score = fundamental_analysis["score"] + consistency_analysis["score"]
        max_possible_score = 10

        # Add margin of safety analysis if we have both intrinsic value and current price
        margin_of_safety = None
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        if intrinsic_value and market_cap:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap

            # Add to score if there's a good margin of safety (>30%)
            if margin_of_safety > 0.3:
                total_score += 2
                max_possible_score += 2

        # Generate trading signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        # Combine all analysis results
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "market_cap": market_cap,
            "margin_of_safety": margin_of_safety,
        }

        progress.update_status("warren_buffett_agent", ticker, "Generating Buffett analysis")
        buffett_output = generate_buffett_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        # Store analysis in consistent format with other agents
        buffett_analysis[ticker] = {
            "signal": buffett_output.signal,
            "confidence": buffett_output.confidence,
            "reasoning": buffett_output.reasoning,
        }

        progress.update_status("warren_buffett_agent", ticker, "Done")

    # Create the message
    message = HumanMessage(content=json.dumps(buffett_analysis), name="warren_buffett_agent")

    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(buffett_analysis, "Warren Buffett Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["warren_buffett_agent"] = buffett_analysis

    return {"messages": [message], "data": state["data"]}


def analyze_fundamentals(metrics: list) -> dict[str, any]:
    """Analyze company fundamentals based on Buffett's criteria."""
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}

    # Get latest metrics
    latest_metrics = metrics[0]

    score = 0
    reasoning = []

    # Check ROE (Return on Equity)
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:  # 15% ROE threshold
        score += 2
        reasoning.append(f"Strong ROE of {latest_metrics.return_on_equity:.1%}")
    elif latest_metrics.return_on_equity:
        reasoning.append(f"Weak ROE of {latest_metrics.return_on_equity:.1%}")
    else:
        reasoning.append("ROE data not available")

    # Check Debt to Equity
    if latest_metrics.debt_to_equity and latest_metrics.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif latest_metrics.debt_to_equity:
        reasoning.append(f"High debt to equity ratio of {latest_metrics.debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    # Check Operating Margin
    if latest_metrics.operating_margin and latest_metrics.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif latest_metrics.operating_margin:
        reasoning.append(f"Weak operating margin of {latest_metrics.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    # Check Current Ratio
    if latest_metrics.current_ratio and latest_metrics.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif latest_metrics.current_ratio:
        reasoning.append(f"Weak liquidity with current ratio of {latest_metrics.current_ratio:.1f}")
    else:
        reasoning.append("Current ratio data not available")

    return {"score": score, "details": "; ".join(reasoning), "metrics": latest_metrics.model_dump()}


def analyze_consistency(financial_line_items: list) -> dict[str, any]:
    """Analyze earnings consistency and growth."""
    if len(financial_line_items) < 4:  # Need at least 4 periods for trend analysis
        return {"score": 0, "details": "Insufficient historical data"}

    score = 0
    reasoning = []

    # Check earnings growth trend
    earnings_values = [item.net_income for item in financial_line_items if item.net_income]
    if len(earnings_values) >= 4:
        earnings_growth = all(earnings_values[i] > earnings_values[i + 1] for i in range(len(earnings_values) - 1))

        if earnings_growth:
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")

        # Calculate growth rate
        if len(earnings_values) >= 2:
            growth_rate = (earnings_values[0] - earnings_values[-1]) / abs(earnings_values[-1])
            reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {
        "score": score,
        "details": "; ".join(reasoning),
    }


def calculate_owner_earnings(financial_line_items: list) -> dict[str, any]:
    """Calculate owner earnings (Buffett's preferred measure of true earnings power).
    Owner Earnings = Net Income + Depreciation - Maintenance CapEx"""
    if not financial_line_items or len(financial_line_items) < 1:
        return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

    latest = financial_line_items[0]

    # Get required components
    net_income = latest.net_income
    depreciation = latest.depreciation_and_amortization
    capex = latest.capital_expenditure

    if not all([net_income, depreciation, capex]):
        return {"owner_earnings": None, "details": ["Missing components for owner earnings calculation"]}

    # Estimate maintenance capex (typically 70-80% of total capex)
    maintenance_capex = capex * 0.75

    owner_earnings = net_income + depreciation - maintenance_capex

    return {
        "owner_earnings": owner_earnings,
        "components": {"net_income": net_income, "depreciation": depreciation, "maintenance_capex": maintenance_capex},
        "details": ["Owner earnings calculated successfully"],
    }


def calculate_intrinsic_value(financial_line_items: list) -> dict[str, any]:
    """Calculate intrinsic value using DCF with owner earnings."""
    if not financial_line_items:
        return {"value": None, "details": ["Insufficient data for valuation"]}

    # Calculate owner earnings
    earnings_data = calculate_owner_earnings(financial_line_items)
    if not earnings_data["owner_earnings"]:
        return {"value": None, "details": earnings_data["details"]}

    owner_earnings = earnings_data["owner_earnings"]

    # Get current market data
    latest_financial_line_items = financial_line_items[0]
    shares_outstanding = latest_financial_line_items.outstanding_shares

    if not shares_outstanding:
        return {"value": None, "details": ["Missing shares outstanding data"]}

    # Buffett's DCF assumptions
    growth_rate = 0.05  # Conservative 5% growth
    discount_rate = 0.09  # Typical 9% discount rate
    terminal_multiple = 12  # Conservative exit multiple
    projection_years = 10

    # Calculate future value
    future_value = 0
    for year in range(1, projection_years + 1):
        future_earnings = owner_earnings * (1 + growth_rate) ** year
        present_value = future_earnings / (1 + discount_rate) ** year
        future_value += present_value

    # Add terminal value
    terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) / (1 + discount_rate) ** projection_years
    intrinsic_value = future_value + terminal_value

    return {
        "intrinsic_value": intrinsic_value,
        "owner_earnings": owner_earnings,
        "assumptions": {
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "terminal_multiple": terminal_multiple,
            "projection_years": projection_years,
        },
        "details": ["Intrinsic value calculated using DCF model with owner earnings"],
    }


def generate_buffett_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> WarrenBuffettSignal:
    """Get investment decision from LLM with Buffett's principles, enhanced for Perplexity Sonar Reasoning."""
    
    # Check if using Perplexity
    using_perplexity = "perplexity" in model_provider.lower() or "sonar" in model_name.lower()
    
    # Create a more detailed system prompt that takes advantage of Perplexity's strengths
    if using_perplexity:
        system_prompt = """You are a Warren Buffett AI agent with enhanced reasoning capabilities. 
        Your task is to deeply analyze companies using Warren Buffett's investment principles and make investment decisions.
        
        Follow these principles with rigorous reasoning at each step:
        
        1. Circle of Competence: 
           - Analyze if the business model is easy to understand
           - Determine if the company operates in a predictable industry
           - Assess if future cash flows can be reasonably estimated
        
        2. Margin of Safety:
           - Calculate a conservative intrinsic value
           - Compare to current market price with detailed numerical analysis
           - Only recommend buying when trading at least 30% below intrinsic value
        
        3. Economic Moat Analysis:
           - Identify specific competitive advantages (brand, patents, network effects, cost advantages)
           - Quantify how these advantages translate to superior returns
           - Evaluate sustainability of the moat over the next decade
        
        4. Management Assessment:
           - Analyze capital allocation decisions over 5+ years
           - Evaluate shareholder-friendly actions (buybacks, dividends)
           - Assess management integrity and ownership alignment
        
        5. Financial Health:
           - Analyze debt levels relative to industry averages
           - Evaluate consistency of return on equity and return on invested capital
           - Check for accounting red flags or aggressive practices
        
        For your analysis:
        - Show step-by-step reasoning for each principle
        - Cite specific metrics and numbers from the provided data
        - Weigh competing factors explicitly
        - Be willing to say "too hard" for businesses outside your circle of competence
        - Apply second-level thinking to identify nuances others might miss
        
        Return a JSON object with:
        - signal: "bullish", "bearish", or "neutral"
        - confidence: numerical confidence level (0-100)
        - reasoning: detailed explanation with step-by-step analysis
        """
    else:
        # Original system prompt for other models
        system_prompt = """You are a Warren Buffett AI agent. Decide on investment signals based on Warren Buffett's principles:

        Circle of Competence: Only invest in businesses you understand
        Margin of Safety: Buy well below intrinsic value
        Economic Moat: Prefer companies with lasting advantages
        Quality Management: Look for conservative, shareholder-oriented teams
        Financial Strength: Low debt, strong returns on equity
        Long-term Perspective: Invest in businesses, not just stocks

        Rules:
        - Buy only if margin of safety > 30%
        - Focus on owner earnings and intrinsic value
        - Prefer consistent earnings growth
        - Avoid high debt or poor management
        - Hold good businesses long term
        - Sell when fundamentals deteriorate or the valuation is too high
        """
    
    # Create the prompt template with the enhanced system prompt
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            system_prompt
        ),
        (
            "human",
            """Based on the following data, create the investment signal as Warren Buffett would.

            Analysis Data for {ticker}:
            {analysis_data}

            Return the trading signal in the following JSON format:
            {{
              "signal": "bullish/bearish/neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """
        ),
    ])

    # Generate the prompt
    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2), 
        "ticker": ticker
    })

    # Create default factory for WarrenBuffettSignal
    def create_default_warren_buffett_signal():
        return WarrenBuffettSignal(signal="neutral", confidence=0.0, reasoning="Error in analysis, defaulting to neutral")

    return call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=WarrenBuffettSignal, 
        agent_name="warren_buffett_agent", 
        default_factory=create_default_warren_buffett_signal,
    )
4. .env.example
Copy# For running LLMs hosted by openai (gpt-4o, gpt-4o-mini, etc.)
# Get your OpenAI API key from https://platform.openai.com/
OPENAI_API_KEY=your-openai-api-key

# For running LLMs hosted by groq (deepseek, llama3, etc.)
# Get your Groq API key from https://groq.com/
GROQ_API_KEY=your-groq-api-key

# For running LLMs hosted by anthropic (claude-3-5-sonnet, claude-3-opus, claude-3-5-haiku)
# Get your Anthropic API key from https://anthropic.com/
ANTHROPIC_API_KEY=your-anthropic-api-key

# For running Perplexity Sonar Reasoning models (sonar-small-online, sonar-medium-online, sonar-large-online)
# Get your Perplexity API key from https://www.perplexity.ai/api
PERPLEXITY_API_KEY=your-perplexity-api-key

# For getting financial data to power the hedge fund
# Get your Financial Datasets API key from https://financialdatasets.ai/
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
5. pyproject.toml (updated with dashboard requirements)
tomlCopy[tool.poetry]
name = "ai-hedge-fund"
version = "0.1.0"
description = "An AI-powered hedge fund that uses multiple agents to make trading decisions"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
packages = [
    { include = "src", from = "." }
]
[tool.poetry.dependencies]
python = "^3.9"
langchain = "0.3.0"
langchain-anthropic = "0.3.5"
langchain-groq = "0.2.3"
langchain-openai = "0.3"
langgraph = "0.2.56"
pandas = "^2.1.0"
numpy = "^1.24.0"
python-dotenv = "1.0.0"
matplotlib = "^3.9.2"
tabulate = "^0.9.0"
colorama = "^0.4.6"
questionary = "^2.1.0"
rich = "^13.9.4"
streamlit = "1.32.0"
altair = "5.2.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.7.0"
isort = "^5.12.0"
flake8 = "^6.1.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 420
target-version = ['py39']
include = '\.pyi?$'