"""
TradingGraph: Simplified orchestrator for the multi-agent trading system.
Directly calls agent functions without LangGraph tool calling complexity.
"""
import os
from typing import Dict
from langchain_openai import ChatOpenAI
from default_config import DEFAULT_CONFIG
from graph_util import TechnicalTools

def safe_str(obj):
    """Safely convert object to string, handling encoding issues"""
    try:
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        elif isinstance(obj, str):
            return obj.encode('utf-8', errors='replace').decode('utf-8')
        else:
            return str(obj).encode('utf-8', errors='replace').decode('utf-8')
    except Exception:
        try:
            return repr(obj)
        except Exception:
            return "Error converting to string"

class TradingGraph:
    """
    Simplified orchestrator for the multi-agent trading system.
    Directly calls agent functions without complex tool calling.
    """
    def __init__(self, config=None):
        # --- Configuration and LLMs ---
        self.config = config if config is not None else DEFAULT_CONFIG.copy()
        
        # Initialize LLMs with proper provider support
        self.agent_llm = self._create_llm(
            model=self.config.get("agent_llm_model", "gpt-4o-mini"),
            temperature=self.config.get("agent_llm_temperature", 0.1)
        )
        self.graph_llm = self._create_llm(
            model=self.config.get("graph_llm_model", "gpt-4o"),
            temperature=self.config.get("graph_llm_temperature", 0.1)
        )
        self.toolkit = TechnicalTools()

    def _create_llm(self, model="gpt-4o-mini", temperature=0.1):
        """Create LLM instance with proper provider support"""
        try:
            # Check current LLM provider
            llm_provider = os.environ.get("LLM_PROVIDER", "deepseek")
            
            if llm_provider == "deepseek":
                deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
                if deepseek_key and deepseek_key != "your-deepseek-api-key-here":
                    # Use DeepSeek model names
                    deepseek_model = "deepseek-chat" if "gpt" in model.lower() else model
                    return ChatOpenAI(
                        model=deepseek_model,
                        temperature=temperature,
                        api_key=deepseek_key,
                        base_url="https://api.deepseek.com/v1"
                    )
            
            # Default to OpenAI or if DeepSeek is not available
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key and openai_key != "your-openai-api-key-here":
                return ChatOpenAI(
                    model=model,
                    temperature=temperature,
                    api_key=openai_key
                )
            
            # If no valid API key, use default config (may fail)
            print(f"Warning: No valid API key found for LLM provider: {llm_provider}")
            return ChatOpenAI(
                model=model,
                temperature=temperature
            )
            
        except Exception as e:
            error_msg = safe_str(e)
            print(f"Error creating LLM: {error_msg}")
            # Return default config as fallback
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=temperature
            )

    def refresh_llms(self):
        """
        Refresh the LLM objects with the current API key from environment.
        This is called when the API key is updated.
        """
        try:
            # Recreate LLM objects with current environment API key and config values
            self.agent_llm = self._create_llm(
                model=self.config.get("agent_llm_model", "gpt-4o-mini"),
                temperature=self.config.get("agent_llm_temperature", 0.1)
            )
            self.graph_llm = self._create_llm(
                model=self.config.get("graph_llm_model", "gpt-4o"),
                temperature=self.config.get("graph_llm_temperature", 0.1)
            )
            
        except Exception as e:
            error_msg = safe_str(e)
            print(f"Error refreshing LLMs: {error_msg}")

    def analyze(self, data, asset_symbol="BTC"):
        """
        Run the complete trading analysis pipeline using direct function calls
        
        Args:
            data: Market data (DataFrame)
            asset_symbol: Asset symbol for analysis
            
        Returns:
            Dict containing analysis results from all agents
        """
        try:
            # Import agent functions
            from indicator_agent import create_indicator_agent
            from pattern_agent import create_pattern_agent
            from trend_agent import create_trend_agent
            from decision_agent import create_decision_agent

            # Create agent functions
            indicator_agent = create_indicator_agent(self.agent_llm, self.toolkit.get_indicator_tools())
            pattern_agent = create_pattern_agent(self.agent_llm, self.toolkit.get_pattern_tools())
            trend_agent = create_trend_agent(self.agent_llm, self.toolkit.get_trend_tools())
            decision_agent = create_decision_agent(self.graph_llm, self.toolkit.get_decision_tools())

            # Prepare state
            state = {
                "kline_data": data,
                "data": data,
                "asset_symbol": asset_symbol,
                "time_frame": "1d",  # Default timeframe
                "stock_name": asset_symbol,
                "messages": [],
                "indicator_report": "",
                "pattern_report": "",
                "trend_report": "",
                "decision_report": ""
            }

            # Run agents sequentially
            print("Running indicator analysis...")
            state = indicator_agent(state)
            
            print("Running pattern analysis...")
            state = pattern_agent(state)
            
            print("Running trend analysis...")
            state = trend_agent(state)
            
            print("Running decision analysis...")
            state = decision_agent(state)

            return {
                "success": True,
                "final_state": {
                    "indicator_report": state.get("indicator_report", ""),
                    "pattern_report": state.get("pattern_report", ""),
                    "trend_report": state.get("trend_report", ""),
                    "final_trade_decision": state.get("final_trade_decision", ""),
                    "pattern_image": state.get("pattern_image", ""),
                    "trend_image": state.get("trend_image", ""),
                    "pattern_image_filename": state.get("pattern_image_filename", ""),
                    "trend_image_filename": state.get("trend_image_filename", "")
                }
            }
            
        except Exception as e:
            error_msg = f"Analysis failed: {safe_str(e)}"
            print(f"TradingGraph analysis error: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "final_state": {
                    "indicator_report": f"指标分析失败: {error_msg}",
                    "pattern_report": f"形态分析失败: {error_msg}",
                    "trend_report": f"趋势分析失败: {error_msg}",
                    "final_trade_decision": f"决策分析失败: {error_msg}",
                    "pattern_image": "",
                    "trend_image": "",
                    "pattern_image_filename": "",
                    "trend_image_filename": ""
                }
            }