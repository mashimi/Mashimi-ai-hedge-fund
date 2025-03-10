"""
Stock Predictor using Perplexity Sonar Reasoning model.
"""
import os
import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from llm.perplexity_langchain import ChatPerplexity

class PerplexityConfig:
    """Configuration for Perplexity models."""
    MODELS = {
        'reasoning': {
            'name': 'sonar-reasoning',
            'max_tokens': 4096,
            'temperature_range': (0.1, 1.0)
        },
        'medium': {
            'name': 'sonar-medium-online',
            'max_tokens': 4096,
            'temperature_range': (0.1, 1.0)
        },
        'small': {
            'name': 'sonar-small-online',
            'max_tokens': 4096,
            'temperature_range': (0.1, 1.0)
        },
        'large': {
            'name': 'sonar-large-online',
            'max_tokens': 4096,
            'temperature_range': (0.1, 1.0)
        }
    }

class StockPredictor:
    """
    Stock predictor using Perplexity Sonar Reasoning model.
    
    This class provides enhanced stock analysis capabilities using Perplexity's
    Sonar Reasoning model, which is optimized for complex reasoning tasks.
    """
    
    def __init__(self, model_type='reasoning', temperature=0.5, max_tokens=None, timeout=120.0, max_retries=5):
        """
        Initialize the StockPredictor.
        
        Args:
            model_type: Type of Perplexity model to use ('reasoning', 'medium', 'small', 'large')
            temperature: Temperature for sampling (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate (default: None)
            timeout: Timeout for API requests in seconds (default: 120.0)
            max_retries: Maximum number of retries for failed requests (default: 5)
        """
        if model_type not in PerplexityConfig.MODELS:
            raise ValueError(f"Invalid model type. Choose from: {list(PerplexityConfig.MODELS.keys())}")
            
        model_config = PerplexityConfig.MODELS[model_type]
        
        # Get API key from environment variables
        api_key = os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY")
        if not api_key:
            raise ValueError("Perplexity API key not found. Please set PERPLEXITY_API_KEY in your .env file.")
        
        # Validate temperature
        temp_range = model_config['temperature_range']
        if temperature < temp_range[0] or temperature > temp_range[1]:
            print(f"Warning: Temperature {temperature} is outside recommended range {temp_range}. Clamping to range.")
            temperature = max(temp_range[0], min(temperature, temp_range[1]))
        
        # Initialize the LLM
        self.llm = ChatPerplexity(
            model=model_config['name'],
            temperature=temperature,
            max_tokens=max_tokens or model_config['max_tokens'],
            pplx_api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Build the analysis chains
        self.chains = self._build_chains()
    
    def _build_chains(self):
        """Build the analysis chains."""
        # Fundamental analysis template
        fundamental_template = ChatPromptTemplate.from_messages([
            ("system", """You are a world-class financial analyst with expertise in fundamental analysis.
            Analyze the provided company data thoroughly and provide detailed insights.
            Focus on financial health, growth prospects, competitive position, and valuation.
            Be specific, cite numbers, and provide a clear investment recommendation.
            
            IMPORTANT: Return ONLY valid JSON without any additional text, explanations, or thinking process."""),
            ("human", """Analyze this company: {ticker}
            
            Financial Data:
            {financial_data}
            
            Provide a detailed fundamental analysis with the following structure:
            1. Financial Health Assessment
            2. Growth Analysis
            3. Competitive Position
            4. Valuation Analysis
            5. Investment Recommendation (Bullish/Bearish/Neutral with confidence percentage)
            
            Return your analysis in JSON format with the following structure:
            {{
                "financial_health": "detailed analysis",
                "growth_analysis": "detailed analysis",
                "competitive_position": "detailed analysis",
                "valuation": "detailed analysis",
                "recommendation": {{
                    "signal": "bullish/bearish/neutral",
                    "confidence": float (0-100),
                    "reasoning": "detailed reasoning"
                }}
            }}
            """)
        ])
        
        # Technical analysis template
        technical_template = ChatPromptTemplate.from_messages([
            ("system", """You are a world-class technical analyst with expertise in chart patterns, indicators, and market trends.
            Analyze the provided price and volume data thoroughly and provide detailed technical insights.
            Focus on trend analysis, support/resistance levels, momentum indicators, and volume analysis.
            Be specific, cite numbers, and provide a clear trading recommendation.
            
            IMPORTANT: Return ONLY valid JSON without any additional text, explanations, or thinking process."""),
            ("human", """Analyze the technical indicators for: {ticker}
            
            Price and Volume Data:
            {price_data}
            
            Technical Indicators:
            {technical_indicators}
            
            Provide a detailed technical analysis with the following structure:
            1. Trend Analysis
            2. Support/Resistance Levels
            3. Momentum Indicators
            4. Volume Analysis
            5. Trading Recommendation (Bullish/Bearish/Neutral with confidence percentage)
            
            Return your analysis in JSON format with the following structure:
            {{
                "trend_analysis": "detailed analysis",
                "support_resistance": "detailed analysis",
                "momentum": "detailed analysis",
                "volume_analysis": "detailed analysis",
                "recommendation": {{
                    "signal": "bullish/bearish/neutral",
                    "confidence": float (0-100),
                    "reasoning": "detailed reasoning"
                }}
            }}
            """)
        ])
        
        # Sentiment analysis template
        sentiment_template = ChatPromptTemplate.from_messages([
            ("system", """You are a world-class sentiment analyst with expertise in market psychology, news analysis, and social media trends.
            Analyze the provided sentiment data thoroughly and provide detailed insights.
            Focus on news sentiment, social media trends, analyst opinions, and institutional activity.
            Be specific, cite examples, and provide a clear sentiment recommendation.
            
            IMPORTANT: Return ONLY valid JSON without any additional text, explanations, or thinking process."""),
            ("human", """Analyze the sentiment for: {ticker}
            
            News and Social Media Data:
            {sentiment_data}
            
            Provide a detailed sentiment analysis with the following structure:
            1. News Sentiment
            2. Social Media Trends
            3. Analyst Opinions
            4. Institutional Activity
            5. Sentiment Recommendation (Bullish/Bearish/Neutral with confidence percentage)
            
            Return your analysis in JSON format with the following structure:
            {{
                "news_sentiment": "detailed analysis",
                "social_media": "detailed analysis",
                "analyst_opinions": "detailed analysis",
                "institutional_activity": "detailed analysis",
                "recommendation": {{
                    "signal": "bullish/bearish/neutral",
                    "confidence": float (0-100),
                    "reasoning": "detailed reasoning"
                }}
            }}
            """)
        ])
        
        # Final recommendation template
        recommendation_template = ChatPromptTemplate.from_messages([
            ("system", """You are a world-class investment advisor with expertise in integrating fundamental, technical, and sentiment analysis.
            Analyze the provided analyses thoroughly and provide a comprehensive investment recommendation.
            Focus on synthesizing the different perspectives, identifying conflicts, and providing a balanced view.
            Be specific, cite the most important factors, and provide a clear final recommendation.
            
            IMPORTANT: Return ONLY valid JSON without any additional text, explanations, or thinking process."""),
            ("human", """Synthesize these analyses for: {ticker}
            
            Fundamental Analysis:
            {fundamental_analysis}
            
            Technical Analysis:
            {technical_analysis}
            
            Sentiment Analysis:
            {sentiment_analysis}
            
            Provide a comprehensive investment recommendation with the following structure:
            1. Summary of Key Points
            2. Areas of Agreement
            3. Areas of Conflict
            4. Risk Assessment
            5. Final Recommendation (Bullish/Bearish/Neutral with confidence percentage)
            
            Return your recommendation in JSON format with the following structure:
            {{
                "summary": "summary of key points",
                "agreements": "areas where analyses agree",
                "conflicts": "areas where analyses conflict",
                "risks": "key risks to consider",
                "recommendation": {{
                    "signal": "bullish/bearish/neutral",
                    "confidence": float (0-100),
                    "reasoning": "detailed reasoning",
                    "time_horizon": "short/medium/long term"
                }}
            }}
            """)
        ])
        
        return {
            "fundamental": fundamental_template,
            "technical": technical_template,
            "sentiment": sentiment_template,
            "recommendation": recommendation_template
        }
    
    def analyze_fundamentals(self, ticker: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze company fundamentals.
        
        Args:
            ticker: Stock ticker symbol
            financial_data: Dictionary containing financial data
            
        Returns:
            Dictionary containing fundamental analysis
        """
        prompt = self.chains["fundamental"].invoke({
            "ticker": ticker,
            "financial_data": financial_data
        })
        
        response = self.llm.invoke(prompt)
        content = response.content
        
        # Extract JSON from the response
        json_content = self._extract_json_from_response(content)
        return json_content
    
    def analyze_technicals(self, ticker: str, price_data: Dict[str, Any], technical_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze technical indicators.
        
        Args:
            ticker: Stock ticker symbol
            price_data: Dictionary containing price and volume data
            technical_indicators: Dictionary containing technical indicators
            
        Returns:
            Dictionary containing technical analysis
        """
        prompt = self.chains["technical"].invoke({
            "ticker": ticker,
            "price_data": price_data,
            "technical_indicators": technical_indicators
        })
        
        response = self.llm.invoke(prompt)
        content = response.content
        
        # Extract JSON from the response
        json_content = self._extract_json_from_response(content)
        return json_content
    
    def analyze_sentiment(self, ticker: str, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market sentiment.
        
        Args:
            ticker: Stock ticker symbol
            sentiment_data: Dictionary containing sentiment data
            
        Returns:
            Dictionary containing sentiment analysis
        """
        prompt = self.chains["sentiment"].invoke({
            "ticker": ticker,
            "sentiment_data": sentiment_data
        })
        
        response = self.llm.invoke(prompt)
        content = response.content
        
        # Extract JSON from the response
        json_content = self._extract_json_from_response(content)
        return json_content
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON from a response that might contain other text.
        
        Args:
            response: Response string that might contain JSON
            
        Returns:
            Extracted JSON string
        """
        # Look for JSON in code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(json_pattern, response)
        if matches:
            return matches[0].strip()
        
        # Look for JSON between curly braces
        json_pattern = r"\{[\s\S]*\}"
        matches = re.findall(json_pattern, response)
        if matches:
            return matches[0].strip()
        
        # If no JSON found, return the original response
        return response
    
    def get_recommendation(
        self, 
        ticker: str, 
        fundamental_analysis: Dict[str, Any], 
        technical_analysis: Dict[str, Any], 
        sentiment_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get final investment recommendation.
        
        Args:
            ticker: Stock ticker symbol
            fundamental_analysis: Dictionary containing fundamental analysis
            technical_analysis: Dictionary containing technical analysis
            sentiment_analysis: Dictionary containing sentiment analysis
            
        Returns:
            Dictionary containing final recommendation
        """
        prompt = self.chains["recommendation"].invoke({
            "ticker": ticker,
            "fundamental_analysis": fundamental_analysis,
            "technical_analysis": technical_analysis,
            "sentiment_analysis": sentiment_analysis
        })
        
        response = self.llm.invoke(prompt)
        content = response.content
        
        # Extract JSON from the response
        json_content = self._extract_json_from_response(content)
        return json_content
    
    def analyze_ticker(self, ticker: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a ticker.
        
        Args:
            ticker: Stock ticker symbol
            data: Dictionary containing all necessary data
                - financial_data: Financial data
                - price_data: Price and volume data
                - technical_indicators: Technical indicators
                - sentiment_data: Sentiment data
                
        Returns:
            Dictionary containing comprehensive analysis and recommendation
        """
        # Extract data
        financial_data = data.get("financial_data", {})
        price_data = data.get("price_data", {})
        technical_indicators = data.get("technical_indicators", {})
        sentiment_data = data.get("sentiment_data", {})
        
        # Perform analyses
        try:
            fundamental_analysis = self.analyze_fundamentals(ticker, financial_data)
        except Exception as e:
            print(f"Error analyzing fundamentals: {str(e)}")
            fundamental_analysis = '{"financial_health": "No data available", "recommendation": {"signal": "neutral", "confidence": 50, "reasoning": "Insufficient data"}}'
            
        try:
            technical_analysis = self.analyze_technicals(ticker, price_data, technical_indicators)
        except Exception as e:
            print(f"Error analyzing technicals: {str(e)}")
            technical_analysis = '{"trend_analysis": "No data available", "recommendation": {"signal": "neutral", "confidence": 50, "reasoning": "Insufficient data"}}'
            
        try:
            sentiment_analysis = self.analyze_sentiment(ticker, sentiment_data)
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            sentiment_analysis = '{"news_sentiment": "No data available", "recommendation": {"signal": "neutral", "confidence": 50, "reasoning": "Insufficient data"}}'
        
        # Get final recommendation
        try:
            recommendation = self.get_recommendation(
                ticker, 
                fundamental_analysis, 
                technical_analysis, 
                sentiment_analysis
            )
        except Exception as e:
            print(f"Error getting recommendation: {str(e)}")
            recommendation = json.dumps({
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
            })
        
        return {
            "ticker": ticker,
            "fundamental_analysis": fundamental_analysis,
            "technical_analysis": technical_analysis,
            "sentiment_analysis": sentiment_analysis,
            "recommendation": recommendation
        }
