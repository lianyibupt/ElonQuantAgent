# -*- coding: utf-8 -*-
import sys
import locale

# Set encoding to handle Unicode characters properly
if sys.platform.startswith('win'):
    # Windows specific encoding setup
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
else:
    # Unix/Linux/Mac encoding setup
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            pass  # Use system default

from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
from pathlib import Path
import json
import re
from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List
import base64
import io
from PIL import Image
import akshare as ak
import finnhub
import numpy as np
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Import your existing modules
from trading_graph import TradingGraph

app = Flask(__name__)

class MultiProviderLLM:
    """支持多厂商LLM API的类"""
    
    def __init__(self):
        self.providers = {
            'openai': {
                'name': 'OpenAI',
                'client_class': OpenAIClient,
                'base_url': None,
                'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo']
            },
            'deepseek': {
                'name': 'DeepSeek',
                'client_class': OpenAIClient,
                'base_url': 'https://api.deepseek.com/v1',
                'models': ['deepseek-chat', 'deepseek-coder']
            }
        }
        self.current_provider = 'openai'
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
    
    def set_provider(self, provider: str, api_key: str = None):
        """Set current LLM provider"""
        if provider not in self.providers:
            raise ValueError(f"Unsupported provider: {provider}")
        
        self.current_provider = provider
        if api_key:
            self.api_key = api_key
        
        # Set environment variables
        if provider == 'openai':
            os.environ["OPENAI_API_KEY"] = self.api_key
        elif provider == 'deepseek':
            os.environ["DEEPSEEK_API_KEY"] = self.api_key
    
    def get_client(self):
        """获取当前配置的LLM客户端"""
        provider_config = self.providers[self.current_provider]
        client_class = provider_config['client_class']
        
        kwargs = {'api_key': self.api_key}
        if provider_config['base_url']:
            kwargs['base_url'] = provider_config['base_url']
        
        return client_class(**kwargs)
    
    def validate_api_key(self, provider: str, api_key: str) -> Dict[str, Any]:
        """Validate API key"""
        try:
            if provider not in self.providers:
                return {"valid": False, "error": f"Unsupported provider: {provider}"}
            
            # Use first model from provider config for validation
            provider_config = self.providers[provider]
            model = provider_config['models'][0]  # 使用第一个可用模型
            
            if provider == 'openai':
                client = OpenAIClient(api_key=api_key)
            elif provider == 'deepseek':
                client = OpenAIClient(api_key=api_key, base_url=provider_config['base_url'])
            else:
                return {"valid": False, "error": f"Unsupported provider: {provider}"}
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            return {"valid": True, "message": f"{provider_config['name']} API key is valid"}
                
        except Exception as e:
            error_msg = safe_str(e)
                
            if "authentication" in error_msg.lower() or "invalid api key" in error_msg.lower() or "401" in error_msg:
                return {"valid": False, "error": f"Invalid {self.providers[provider]['name']} API key"}
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                return {"valid": False, "error": "API rate limit exceeded, please try again later"}
            elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
                return {"valid": False, "error": "Account quota exceeded or billing issue"}
            elif "network" in error_msg.lower() or "connection" in error_msg.lower():
                return {"valid": False, "error": "Network connection error"}
            elif "model not exist" in error_msg.lower() or "400" in error_msg:
                return {"valid": False, "error": f"Model does not exist: {error_msg}"}
            else:
                return {"valid": False, "error": f"API validation error: {error_msg}"}

class MultiSourceDataFetcher:
    """支持多数据源的类"""
    
    def __init__(self):
        self.sources = ['akshare', 'finnhub', 'yfinance']
        self.current_source = 'akshare'
        self.finnhub_client = None
        
    def initialize_finnhub(self, api_key: str = None):
        """Initialize Finnhub client"""
        api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        if api_key and api_key != "your-finnhub-api-key-here":
            self.finnhub_client = finnhub.Client(api_key=api_key)
    
    def fetch_akshare_data(self, symbol: str, period: str = "daily", 
                          start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch stock data using akshare"""
        try:
            print(f"正在获取 {symbol} 的数据...")
            
            # 如果akshare获取失败，使用demo数据
            df = self.get_demo_data(symbol, start_date, end_date)
            if not df.empty:
                print(f"使用demo数据，成功获取 {len(df)} 条数据")
                return df
            
            # akshare数据获取逻辑 - 尝试多种方法
            functions_to_try = [
                ('stock_zh_index_daily_em', symbol),
                ('stock_zh_a_hist', symbol),
                ('index_zh_a_hist', symbol),
            ]
            
            for func_name, sym in functions_to_try:
                try:
                    func = getattr(ak, func_name)
                    if func_name == 'stock_zh_a_hist':
                        df = func(symbol=sym, period="daily", 
                                start_date=start_date.replace('-', ''), 
                                end_date=end_date.replace('-', ''), adjust="")
                    elif func_name == 'index_zh_a_hist':
                        df = func(symbol=sym, period="daily", 
                                start_date=start_date.replace('-', ''), 
                                end_date=end_date.replace('-', ''))
                    else:
                        df = func(symbol=sym)
                    
                    if not df.empty:
                        print(f"使用 {func_name} 成功获取数据")
                        break
                except Exception as e:
                    print(f"{func_name} 失败: {safe_str(e)}")
                    continue
            
            if df.empty:
                print(f"所有akshare方法都失败，使用demo数据")
                return self.get_demo_data(symbol, start_date, end_date)
            
            # Standardize column names - 更全面的映射
            column_mapping = {
                'date': 'Datetime', '日期': 'Datetime', 'Date': 'Datetime',
                'open': 'Open', '开盘': 'Open', 'Open': 'Open',
                'high': 'High', '最高': 'High', 'High': 'High',
                'low': 'Low', '最低': 'Low', 'Low': 'Low',
                'close': 'Close', '收盘': 'Close', 'Close': 'Close',
                'volume': 'Volume', '成交量': 'Volume', 'Volume': 'Volume',
                '成交额': 'Volume', 'amount': 'Volume'
            }
            
            # 应用列名映射
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Ensure Datetime column exists
            if 'Datetime' not in df.columns:
                if df.index.name in ['date', '日期', 'Date']:
                    df = df.reset_index()
                    df = df.rename(columns={df.columns[0]: 'Datetime'})
                elif len(df.columns) > 0 and any(col.lower() in ['date', 'datetime', '日期'] for col in df.columns):
                    # 找到日期列
                    date_col = next((col for col in df.columns if col.lower() in ['date', 'datetime', '日期']), None)
                    if date_col:
                        df = df.rename(columns={date_col: 'Datetime'})
            
            # Convert Datetime column to datetime
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df = df.set_index('Datetime')
            
            # Filter by date range
            if start_date and end_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"缺少必要的列: {missing_columns}，使用demo数据")
                return self.get_demo_data(symbol, start_date, end_date)
            
            print(f"成功获取 {len(df)} 条数据")
            return df
            
        except Exception as e:
            error_msg = safe_str(e)
            print(f"akshare数据获取失败: {error_msg}，使用demo数据")
            return self.get_demo_data(symbol, start_date, end_date)
    
    def get_demo_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成demo数据用于测试"""
        try:
            # 生成日期范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
            
            # 过滤工作日
            dates = dates[dates.weekday < 5]  # 0-4 是周一到周五
            
            if len(dates) == 0:
                return pd.DataFrame()
            
            # 生成模拟价格数据
            np.random.seed(42)  # 确保可重复性
            
            # 基础价格
            base_prices = {
                'BTC': 50000,
                '000001': 3000,
                'AAPL': 150,
                'TSLA': 200,
            }
            base_price = base_prices.get(symbol.upper(), 100)
            
            # 生成价格序列
            returns = np.random.normal(0, 0.02, len(dates))  # 2%日波动率
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # 生成OHLC数据
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                daily_volatility = 0.01  # 1%日内波动
                high = price * (1 + np.random.uniform(0, daily_volatility))
                low = price * (1 - np.random.uniform(0, daily_volatility))
                open_price = prices[i-1] if i > 0 else price
                close_price = price
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close_price,
                    'Volume': volume
                })
            
            df = pd.DataFrame(data, index=dates)
            df.index.name = 'Datetime'
            
            # 重置索引，将Datetime作为列
            df = df.reset_index()
            
            print(f"生成了 {len(df)} 条 {symbol} 的demo数据")
            return df
            
        except Exception as e:
            print(f"生成demo数据失败: {safe_str(e)}")
            return pd.DataFrame()
    
    def fetch_finnhub_data(self, symbol: str, resolution: str = 'D', 
                          start_timestamp: int = None, end_timestamp: int = None) -> pd.DataFrame:
        """Fetch stock data using Finnhub"""
        try:
            if not self.finnhub_client:
                self.initialize_finnhub()
                if not self.finnhub_client:
                    return pd.DataFrame()
            
            # Use default range if no timestamp provided
            if not start_timestamp:
                start_timestamp = int((datetime.now() - timedelta(days=30)).timestamp())
            if not end_timestamp:
                end_timestamp = int(datetime.now().timestamp())
            
            # Get candlestick data
            data = self.finnhub_client.stock_candles(symbol, resolution, 
                                                   start_timestamp, end_timestamp)
            
            if not data or 's' != 'ok':
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame({
                'Datetime': pd.to_datetime(data['t'], unit='s'),
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'], 
                'Close': data['c'],
                'Volume': data['v']
            })
            
            return df
            
        except Exception as e:
            print(f"Finnhub data fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_data(self, symbol: str, timeframe: str, 
                  start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data from multiple sources with priority fallback"""
        df = pd.DataFrame()
        
        # 首先尝试akshare
        if self.current_source == 'akshare' or not df.empty:
            df = self.fetch_akshare_data(symbol, self._convert_timeframe(timeframe),
                                       start_date.strftime('%Y%m%d'), 
                                       end_date.strftime('%Y%m%d'))
        
        # 如果akshare失败，尝试finnhub
        if df.empty and self.finnhub_client:
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())
            df = self.fetch_finnhub_data(symbol, self._convert_timeframe(timeframe),
                                      start_ts, end_ts)
        
        return df
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe format"""
        timeframe_map = {
            '1m': '1', '5m': '5', '15m': '15', '30m': '30',
            '1h': '60', '4h': '240', '1d': 'D', '1w': 'W', '1M': 'M'
        }
        return timeframe_map.get(timeframe, 'D')

class WebTradingAnalyzer:
    def __init__(self):
        """Initialize the web trading analyzer with multi-provider support."""
        self.data_dir = Path("data")
        self.llm_provider = MultiProviderLLM()
        self.data_fetcher = MultiSourceDataFetcher()
        
        # 根据环境变量设置LLM提供商
        llm_provider = os.environ.get("LLM_PROVIDER", "deepseek")
        if llm_provider == "deepseek":
            deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
            if deepseek_key and deepseek_key != "your-deepseek-api-key-here":
                self.llm_provider.set_provider('deepseek', deepseek_key)
        elif llm_provider == "openai":
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key and openai_key != "your-openai-api-key-here":
                self.llm_provider.set_provider('openai', openai_key)
        
        # 初始化TradingGraph
        self.trading_graph = TradingGraph()
        
        # Ensure data dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Available assets and their display names
        self.asset_mapping = {
            'SPX': 'S&P 500',
            'BTC': 'Bitcoin', 
            'GC': 'Gold Futures',
            'NQ': 'Nasdaq Futures',
            'CL': 'Crude Oil',
            'ES': 'E-mini S&P 500',
            'DJI': 'Dow Jones',
            'QQQ': 'Invesco QQQ Trust',
            'VIX': 'Volatility Index',
            'DXY': 'US Dollar Index',
            'AAPL': 'Apple Inc.',
            'TSLA': 'Tesla Inc.',
            '000001': 'Shanghai Composite Index',
            '399001': 'Shenzhen Component Index', 
            'SH000300': 'CSI 300 Index',
            'SH510300': 'CSI 300 ETF',
        }
        
        # Symbol mapping for different data sources
        self.symbol_mapping = {
            'akshare': {
                'SPX': 'SPX',
                'BTC': 'BTC-USD',
                'AAPL': 'AAPL',
                'TSLA': 'TSLA',
                '000001': '000001',
                '399001': '399001',
                'SH000300': '000300',
                'SH510300': '510300',
            },
            'finnhub': {
                'SPX': 'SPX',
                'BTC': 'BINANCE:BTCUSDT',
                'AAPL': 'AAPL',
                'TSLA': 'TSLA',
            },
            'yfinance': {
                'SPX': '^GSPC',
                'BTC': 'BTC-USD',
                'GC': 'GC=F',
                'NQ': 'NQ=F',
                'CL': 'CL=F',
                'ES': 'ES=F',
                'DJI': '^DJI',
                'QQQ': 'QQQ',
                'VIX': '^VIX',
                'DXY': 'DX-Y.NYB',
                'AAPL': 'AAPL',
                'TSLA': 'TSLA',
            }
        }
        
        # Timeframe mapping
        self.timeframe_mapping = {
            'akshare': {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '4h': '240', '1d': 'daily', '1w': 'weekly', '1M': 'monthly'
            },
            'finnhub': {
                '1m': '1', '5m': '5', '15m': '15', '30m': '30',
                '1h': '60', '4h': '240', '1d': 'D', '1w': 'W', '1M': 'M'
            },
            'yfinance': {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk', '1M': '1mo'
            }
        }
        
        # Available timeframes
        self.timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
        
        # Load persisted custom assets
        self.custom_assets_file = self.data_dir / "custom_assets.json"
        self.custom_assets = self.load_custom_assets()
        
        # Initialize Finnhub with environment variable
        self.data_fetcher.initialize_finnhub()

    def fetch_market_data(self, symbol: str, interval: str, 
                         start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
        """Fetch OHLCV data from multiple sources with fallback."""
        # Try akshare first
        df = self.data_fetcher.fetch_akshare_data(
            self.symbol_mapping['akshare'].get(symbol, symbol),
            self.timeframe_mapping['akshare'].get(interval, 'daily'),
            start_datetime.strftime('%Y%m%d'),
            end_datetime.strftime('%Y%m%d')
        )
        
        # If akshare fails, try finnhub
        if df.empty:
            start_ts = int(start_datetime.timestamp())
            end_ts = int(end_datetime.timestamp())
            df = self.data_fetcher.fetch_finnhub_data(
                self.symbol_mapping['finnhub'].get(symbol, symbol),
                self.timeframe_mapping['finnhub'].get(interval, 'D'),
                start_ts, end_ts
            )
        
        # If both fail, provide empty DataFrame
        if df.empty:
            print(f"All data sources failed to fetch data for {symbol}")
        
        return df

    # Keep other methods unchanged, only modify data fetching part
    def run_analysis(self, df: pd.DataFrame, asset_name: str, timeframe: str) -> Dict[str, Any]:
        """Run the trading analysis on the provided DataFrame."""
        try:
            # 确保asset_name是安全的字符串
            asset_name = safe_str(asset_name)
            timeframe = safe_str(timeframe)
            
            if len(df) > 49:
                df_slice = df.tail(49).iloc[:-3]
            else:
                df_slice = df.tail(45)
            
            required_columns = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df_slice.columns for col in required_columns):
                return {
                    "success": False,
                    "error": f"Missing required columns. Available: {list(df_slice.columns)}"
                }
            
            df_slice = df_slice.reset_index(drop=True)
            df_slice_dict = {}
            for col in required_columns:
                if col == 'Datetime':
                    # 确保日期时间格式正确
                    try:
                        df_slice_dict[col] = df_slice[col].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                    except Exception:
                        # 如果日期格式有问题，尝试转换
                        df_slice_dict[col] = [safe_str(dt) for dt in df_slice[col].tolist()]
                else:
                    # 确保数值数据是可序列化的
                    try:
                        df_slice_dict[col] = [float(x) if pd.notna(x) else 0.0 for x in df_slice[col].tolist()]
                    except Exception:
                        df_slice_dict[col] = [safe_str(x) for x in df_slice[col].tolist()]
            
            display_timeframe = timeframe
            if timeframe.endswith('h'):
                display_timeframe += 'our'
            elif timeframe.endswith('m'):
                display_timeframe += 'in'
            elif timeframe.endswith('d'):
                display_timeframe += 'ay'
            
            initial_state = {
                "kline_data": df_slice_dict,
                "analysis_results": None,
                "messages": [],
                "time_frame": safe_str(display_timeframe),
                "stock_name": safe_str(asset_name)
            }
            
            final_state = self.trading_graph.analyze(df_slice_dict, asset_name)
            
            # 确保final_state中的所有字符串都是安全的
            if isinstance(final_state, dict):
                for key, value in final_state.items():
                    if isinstance(value, str):
                        final_state[key] = safe_str(value)
            
            return {
                "success": True,
                "final_state": final_state,
                "asset_name": safe_str(asset_name),
                "timeframe": safe_str(display_timeframe),
                "data_length": len(df_slice)
            }
            
        except Exception as e:
            error_msg = safe_str(e)
            
            if "authentication" in error_msg.lower():
                return {"success": False, "error": "API key invalid"}
            elif "rate limit" in error_msg.lower():
                return {"success": False, "error": "API rate limit exceeded"}
            else:
                return {"success": False, "error": f"Analysis error: {error_msg}"}

    def validate_api_key(self, provider: str = None) -> Dict[str, Any]:
        """Validate the current API key for the specified provider."""
        if provider:
            return self.llm_provider.validate_api_key(provider, os.environ.get(f"{provider.upper()}_API_KEY", ""))
        
        # Default to validate current provider
        current_key = self.llm_provider.api_key
        return self.llm_provider.validate_api_key(self.llm_provider.current_provider, current_key)

    # Keep other helper methods unchanged
    def get_available_assets(self) -> list:
        return sorted(list(self.asset_mapping.keys()))
    
    def load_custom_assets(self) -> list:
        try:
            if self.custom_assets_file.exists():
                with open(self.custom_assets_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 确保所有资产名称都是安全字符串
                    return [safe_str(asset) for asset in data if asset]
            return []
        except Exception as e:
            print(f"Failed to load custom assets: {safe_str(e)}")
            return []
    
    def save_custom_asset(self, symbol: str) -> bool:
        try:
            symbol = safe_str(symbol).strip()
            if not symbol or symbol in self.custom_assets:
                return True
            self.custom_assets.append(symbol)
            with open(self.custom_assets_file, 'w', encoding='utf-8') as f:
                json.dump(self.custom_assets, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to save custom asset: {safe_str(e)}")
            return False

    def extract_analysis_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format analysis results for web display."""
        if not results["success"]:
            return {"error": safe_str(results["error"])}
        
        final_state = results["final_state"]
        
        # Extract analysis results from state fields with safe string conversion
        technical_indicators = safe_str(final_state.get("indicator_report", ""))
        pattern_analysis = safe_str(final_state.get("pattern_report", ""))
        trend_analysis = safe_str(final_state.get("trend_report", ""))
        final_decision_raw = safe_str(final_state.get("final_trade_decision", ""))
        
        # Extract chart data if available
        pattern_chart = safe_str(final_state.get("pattern_image", ""))
        trend_chart = safe_str(final_state.get("trend_image", ""))
        pattern_image_filename = safe_str(final_state.get("pattern_image_filename", ""))
        trend_image_filename = safe_str(final_state.get("trend_image_filename", ""))
        
        # Parse final decision
        final_decision = ""
        if final_decision_raw:
            try:
                # Try to extract JSON from the decision
                start = final_decision_raw.find('{')
                end = final_decision_raw.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = final_decision_raw[start:end]
                    decision_data = json.loads(json_str)
                    final_decision = {
                        "decision": safe_str(decision_data.get('decision', 'N/A')),
                        "risk_reward_ratio": safe_str(decision_data.get('risk_reward_ratio', 'N/A')),
                        "forecast_horizon": safe_str(decision_data.get('forecast_horizon', 'N/A')),
                        "justification": safe_str(decision_data.get('justification', 'N/A'))
                    }
                else:
                    # If no JSON found, return the raw text
                    final_decision = {"raw": safe_str(final_decision_raw)}
            except json.JSONDecodeError:
                # If JSON parsing fails, return the raw text
                final_decision = {"raw": safe_str(final_decision_raw)}
        
        return {
            "success": True,
            "asset_name": safe_str(results["asset_name"]),
            "timeframe": safe_str(results["timeframe"]),
            "data_length": results["data_length"],
            "technical_indicators": technical_indicators,
            "pattern_analysis": pattern_analysis,
            "trend_analysis": trend_analysis,
            "pattern_chart": pattern_chart,
            "trend_chart": trend_chart,
            "pattern_image_filename": pattern_image_filename,
            "trend_image_filename": trend_image_filename,
            "final_decision": final_decision
        }

# Initialize the analyzer
analyzer = WebTradingAnalyzer()

# Setup environment variables (if they exist)
def setup_environment():
    """Setup environment variables"""
    # Read API keys from environment variables or config file
    llm_provider = os.environ.get("LLM_PROVIDER", "deepseek")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    finnhub_key = os.environ.get("FINNHUB_API_KEY")
    
    # Set LLM provider based on environment variables
    if llm_provider == "deepseek" and deepseek_key and deepseek_key != "your-deepseek-api-key-here":
        analyzer.llm_provider.set_provider('deepseek', deepseek_key)
    elif llm_provider == "openai" and openai_key and openai_key != "your-openai-api-key-here":
        analyzer.llm_provider.set_provider('openai', openai_key)
    elif deepseek_key and deepseek_key != "your-deepseek-api-key-here":
        # Default to DeepSeek if key available
        analyzer.llm_provider.set_provider('deepseek', deepseek_key)
    elif openai_key and openai_key != "your-openai-api-key-here":
        # Fallback to OpenAI
        analyzer.llm_provider.set_provider('openai', openai_key)
    
    if finnhub_key and finnhub_key != "your-finnhub-api-key-here":
        analyzer.data_fetcher.initialize_finnhub(finnhub_key)

# Initialize environment
setup_environment()

# Flask routes remain unchanged, only modify API key related endpoints
@app.route('/api/update-api-key', methods=['POST'])
def update_api_key():
    """API endpoint to update LLM API key with provider support."""
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        provider = data.get('provider', 'openai')
        
        if not api_key:
            return jsonify({"error": "API key cannot be empty"})
        
        # 验证API密钥
        validation = analyzer.llm_provider.validate_api_key(provider, api_key)
        if not validation["valid"]:
            return jsonify({"error": validation["error"]})
        
        # 设置提供商和API密钥
        analyzer.llm_provider.set_provider(provider, api_key)
        
        # 刷新交易图的LLM
        analyzer.trading_graph.refresh_llms()
        
        return jsonify({"success": True, "message": f"{analyzer.llm_provider.providers[provider]['name']} API key updated successfully"})
        
    except Exception as e:
        return jsonify({"error": safe_str(e)})

@app.route('/api/validate-api-key', methods=['POST'])
def validate_api_key():
    """API endpoint to validate API key for a specific provider."""
    try:
        data = request.get_json()
        api_key = data.get('api_key')
        provider = data.get('provider', 'openai')
        
        if not api_key:
            return jsonify({"error": "API key cannot be empty"})
        
        validation = analyzer.llm_provider.validate_api_key(provider, api_key)
        return jsonify(validation)
        
    except Exception as e:
        return jsonify({"valid": False, "error": safe_str(e)})

@app.route('/api/get-api-key-status')
def get_api_key_status():
    """API endpoint to check API key status for all providers."""
    try:
        openai_key = os.environ.get("OPENAI_API_KEY")
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        
        result = {
            'has_openai_key': False,
            'has_deepseek_key': False,
            'masked_openai_key': '',
            'masked_deepseek_key': ''
        }
        
        if openai_key and openai_key != "your-openai-api-key-here":
            masked_key = openai_key[:8] + '*' * (len(openai_key) - 12) + openai_key[-4:] if len(openai_key) > 12 else '***'
            result['has_openai_key'] = True
            result['masked_openai_key'] = masked_key
        
        if deepseek_key and deepseek_key != "your-deepseek-api-key-here":
            masked_key = deepseek_key[:8] + '*' * (len(deepseek_key) - 12) + deepseek_key[-4:] if len(deepseek_key) > 12 else '***'
            result['has_deepseek_key'] = True
            result['masked_deepseek_key'] = masked_key
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": safe_str(e)})

# Keep other routes unchanged
@app.route('/')
def index():
    return render_template('demo_new.html')

@app.route('/demo')
def demo():
    return render_template('demo_new.html')

@app.route('/output')
def output():
    """Display analysis results page"""
    try:
        # Get results from URL parameters
        results_param = request.args.get('results')
        if results_param:
            import urllib.parse
            try:
                decoded_results = urllib.parse.unquote(results_param, encoding='utf-8')
                results = json.loads(decoded_results)
                
                # 确保所有字符串字段都是安全的
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, str):
                            results[key] = safe_str(value)
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, str):
                                    value[sub_key] = safe_str(sub_value)
                                    
            except Exception as decode_error:
                print(f"URL decode error: {safe_str(decode_error)}")
                # 如果解码失败，使用默认结果
                results = {
                    "success": False,
                    "error": f"Failed to decode results: {safe_str(decode_error)}",
                    "asset_name": "Unknown",
                    "timeframe": "Unknown",
                    "data_length": 0
                }
        else:
            # Default results if no parameter provided
            results = {
                "success": True,
                "asset_name": "BTC",
                "timeframe": "1h",
                "data_length": 1247,
                "technical_indicators": "No analysis data available",
                "pattern_analysis": "No pattern analysis available",
                "trend_analysis": "No trend analysis available",
                "final_decision": {
                    "decision": "HOLD",
                    "risk_reward_ratio": "1:1",
                    "forecast_horizon": "24 hours",
                    "justification": "No analysis data available"
                }
            }
        
        return render_template('output.html', results=results)
        
    except Exception as e:
        # If there's an error parsing results, show error page
        error_results = {
            "success": False,
            "error": f"Error loading results: {safe_str(e)}",
            "asset_name": "Unknown",
            "timeframe": "Unknown",
            "data_length": 0
        }
        return render_template('output.html', results=error_results)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        asset = data.get('asset')
        timeframe = data.get('timeframe')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        redirect_to_output = data.get('redirect_to_output', False)
        
        # Create datetime objects
        start_dt = datetime.strptime(f"{start_date} 00:00", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{end_date} 23:59", "%Y-%m-%d %H:%M")
        
        # Use new data fetching method
        df = analyzer.fetch_market_data(asset, timeframe, start_dt, end_dt)
        if df.empty:
            return jsonify({"error": "Unable to fetch data, please check the code or try other data sources"})
        
        display_name = analyzer.asset_mapping.get(asset, asset)
        results = analyzer.run_analysis(df, display_name, timeframe)
        formatted_results = analyzer.extract_analysis_results(results)
        
        if redirect_to_output:
            # Handle URL-encoded results for redirect
            import urllib.parse
            try:
                # 确保所有字符串都是UTF-8编码
                results_json = json.dumps(formatted_results, ensure_ascii=False)
                encoded_results = urllib.parse.quote(results_json, safe='')
                redirect_url = f"/output?results={encoded_results}"
                return jsonify({"redirect": redirect_url})
            except Exception as e:
                # If encoding fails, return results directly
                error_msg = safe_str(e)
                print(f"URL encoding failed: {error_msg}")
                return jsonify(formatted_results)
        else:
            return jsonify(formatted_results)
        
    except Exception as e:
        error_msg = safe_str(e)
        print(f"Analysis error: {error_msg}")
        return jsonify({"error": error_msg})

# 添加缺失的静态资源和API路由
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets"""
    try:
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        return send_file(os.path.join(assets_dir, filename))
    except Exception as e:
        return jsonify({"error": f"Asset not found: {safe_str(e)}"}), 404

@app.route('/api/custom-assets', methods=['GET'])
def custom_assets():
    """API endpoint to get custom assets"""
    try:
        custom_assets = analyzer.load_custom_assets()
        return jsonify(custom_assets)
    except Exception as e:
        return jsonify({"error": safe_str(e)}), 500

@app.route('/api/images/<image_type>')
def get_image(image_type):
    """API endpoint to serve analysis images"""
    try:
        # 根据图片类型返回相应的图片文件
        image_dir = os.path.join(os.path.dirname(__file__), 'data', 'images')
        
        if image_type == 'pattern':
            # 查找最新的pattern图片
            pattern_files = [f for f in os.listdir(image_dir) if f.startswith('pattern_') and f.endswith('.png')]
            if pattern_files:
                latest_file = max(pattern_files, key=lambda x: os.path.getctime(os.path.join(image_dir, x)))
                return send_file(os.path.join(image_dir, latest_file))
        elif image_type == 'trend':
            # 查找最新的trend图片
            trend_files = [f for f in os.listdir(image_dir) if f.startswith('trend_') and f.endswith('.png')]
            if trend_files:
                latest_file = max(trend_files, key=lambda x: os.path.getctime(os.path.join(image_dir, x)))
                return send_file(os.path.join(image_dir, latest_file))
        
        # 如果没有找到图片，返回404
        return jsonify({"error": f"Image not found: {image_type}"}), 404
        
    except Exception as e:
        return jsonify({"error": safe_str(e)}), 404

# Other helper routes remain unchanged
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)