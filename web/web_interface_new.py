# -*- coding: utf-8 -*-
import sys
import locale
import hashlib

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
from copy import deepcopy
from datetime import datetime, timedelta, date, timezone
from typing import Dict, Any, Optional, List
import base64
import io
from PIL import Image
import akshare as ak
import numpy as np
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv
import yfinance as yf

from services.account_analysis import list_account_analysis_history, save_account_analysis_artifacts
from services.core_skill_account_bridge import build_core_skill_block
from services.stage2_workspace import (
    PRICE_SOURCE_ANALYSIS_REFRESH,
    build_default_workspace,
    normalize_workspace_payload,
    recalculate_account_state,
    refresh_position_after_analysis,
)

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


def build_portfolio_cache_key(account_state: Optional[Dict[str, Any]], positions: Optional[List[Dict[str, Any]]], candidates: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    payload = {
        "account_state": account_state or {},
        "positions": positions or [],
        "candidates": candidates or [],
    }
    if not payload["account_state"] and not payload["positions"] and not payload["candidates"]:
        return None
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# Import your existing modules
from core.trading_graph import TradingGraph

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCOUNT_ANALYSIS_DIR = Path(_project_root) / 'artifacts' / 'account_analysis'
app = Flask(__name__, template_folder=os.path.join(_project_root, 'templates'))

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
                'models': ['deepseek-v4-flash', 'deepseek-v4-pro']
            },
            'volcengine': {
                'name': 'Volcengine',
                'client_class': OpenAIClient,
                'base_url': 'https://ark-cn-beijing.bytedance.net/api/v3',
                'models': ['ep-20250519162223-96wj4']
            }
        }
        self.current_provider = 'volcengine'
        self.api_key = os.environ.get("VOLCENGINE_API_KEY", "")
    
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
        elif provider == 'volcengine':
            os.environ["VOLCENGINE_API_KEY"] = self.api_key
    
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
            elif provider == 'volcengine':
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
        self.sources = ['akshare', 'yfinance']
        configured_source = os.environ.get("DATA_SOURCE", "akshare")
        configured_source = safe_str(configured_source).strip().lower()
        self.current_source = configured_source if configured_source in self.sources else 'akshare'
    
    def fetch_akshare_data(self, symbol: str, period: str = "daily", 
                          start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch stock data using akshare"""
        try:
            print(f"正在获取 {symbol} 的数据...")
            print(f"  开始日期: {start_date}, 结束日期: {end_date}")
            
            # 先尝试获取真实数据，如果失败再使用demo数据
            df = pd.DataFrame()
            
            # akshare数据获取逻辑 - 尝试多种方法
            # 根据股票代码类型选择不同的函数
            functions_to_try = []
            
            # 美股代码（通常为1-5个字母）
            if re.match(r'^[A-Z]{1,5}$', symbol):
                print(f"  检测到美股代码: {symbol}")
                functions_to_try.extend([
                    ('stock_us_daily', symbol),  # 美股日线数据
                    ('stock_us_spot', symbol),   # 美股实时数据
                ])
            # A股代码（6位数字）
            elif re.match(r'^\d{6}$', symbol):
                print(f"  检测到A股代码: {symbol}")
                functions_to_try.extend([
                    ('stock_zh_a_hist', symbol),
                    ('stock_zh_index_daily_em', symbol),
                ])
            # 指数代码
            elif symbol.startswith('SH') or symbol.startswith('SZ'):
                print(f"  检测到指数代码: {symbol}")
                functions_to_try.extend([
                    ('index_zh_a_hist', symbol),
                    ('stock_zh_index_daily_em', symbol),
                ])
            else:
                print(f"  未知代码格式: {symbol}，尝试所有方法")
                functions_to_try.extend([
                    ('stock_zh_index_daily_em', symbol),
                    ('stock_zh_a_hist', symbol),
                    ('index_zh_a_hist', symbol),
                    ('stock_us_daily', symbol),
                    ('stock_us_spot', symbol),
                ])
            
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
                    elif func_name == 'stock_us_daily':
                        # 美股日线数据
                        df = func(symbol=sym)
                    elif func_name == 'stock_us_spot':
                        # 美股实时数据，可能需要转换为日线
                        df = func()
                        if not df.empty and symbol in df['symbol'].values:
                            df = df[df['symbol'] == symbol]
                    else:
                        df = func(symbol=sym)
                    
                    # 检查df是否为None或空
                    if df is None:
                        print(f"{func_name} 返回None，跳过")
                        continue
                    
                    if not df.empty:
                        print(f"使用 {func_name} 成功获取数据，数据形状: {df.shape}")
                        break
                    else:
                        print(f"{func_name} 返回空DataFrame")
                        
                except Exception as e:
                    error_msg = safe_str(e)
                    print(f"{func_name} 失败: {error_msg}")
                    if "subscriptable" in error_msg:
                        print(f"  详细错误: 函数 {func_name} 可能返回了None")
                    continue
            
            if df.empty:
                print(f"❌ 所有akshare方法都失败,无法获取 {symbol} 的真实数据")
                print(f"   请检查股票代码是否正确,或稍后重试")
                return pd.DataFrame()  # 返回空DataFrame,不生成误导性的模拟数据
            
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
            print(f"  原始列名: {list(df.columns)}")
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            print(f"  映射后列名: {list(df.columns)}")
            
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
                print(f"❌ 数据缺少必要的列: {missing_columns}")
                print(f"   无法进行分析,请检查数据源")
                return pd.DataFrame()  # 返回空DataFrame,不生成误导性的模拟数据
            
            print(f"成功获取 {len(df)} 条数据")
            
            # 添加详细的价格数据日志
            print(f"\n📊 [数据获取] {symbol} 价格数据详情:")
            print(f"  数据形状: {df.shape}")
            if 'Close' in df.columns:
                close_prices = df['Close'].values
                print(f"  收盘价范围: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
                print(f"  最新收盘价: ${close_prices[-1]:.2f}")
                print(f"  前3条收盘价: {close_prices[:3].tolist()}")
                print(f"  后3条收盘价: {close_prices[-3:].tolist()}")
            
            return df
            
        except Exception as e:
            error_msg = safe_str(e)
            print(f"❌ akshare数据获取失败: {error_msg}")
            print(f"   无法获取真实市场数据,请稍后重试")
            return pd.DataFrame()  # 返回空DataFrame,不生成误导性的模拟数据
    

    def fetch_yfinance_data_with_datetime(self, symbol: str, interval: str,
                                          start_datetime: datetime, end_datetime: datetime) -> pd.DataFrame:
        try:
            df = yf.download(
                tickers=symbol,
                start=start_datetime,
                end=end_datetime,
                interval=interval,
                auto_adjust=True,
                prepost=False
            )

            if df is None or df.empty:
                return pd.DataFrame()

            if isinstance(df, pd.Series):
                df = df.to_frame()

            df = df.reset_index()

            if not isinstance(df, pd.DataFrame):
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            column_mapping = {
                'Date': 'Datetime',
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            }

            existing = {old: new for old, new in column_mapping.items() if old in df.columns}
            if existing:
                df = df.rename(columns=existing)

            required = ["Datetime", "Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required):
                return pd.DataFrame()

            df = df[required + (["Volume"] if "Volume" in df.columns else [])]
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            return df
        except Exception as e:
            print(f"yfinance fetch failed: {safe_str(e)}")
            return pd.DataFrame()
    

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
        elif llm_provider == "volcengine":
            volcengine_key = os.environ.get("VOLCENGINE_API_KEY")
            if volcengine_key and volcengine_key != "your-volcengine-api-key-here":
                self.llm_provider.set_provider('volcengine', volcengine_key)
        
        # 初始化TradingGraph
        self.trading_graph = TradingGraph()
        
        # Ensure data dir exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Available assets and their display names
        self.asset_mapping = {
            'SPX': 'S&P 500',
            'BTC': 'Bitcoin', 
            'ETH': 'Ethereum',
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
                'ETH': 'ETH-USD',
                'AAPL': 'AAPL',
                'TSLA': 'TSLA',
                '000001': '000001',
                '399001': '399001',
                'SH000300': '000300',
                'SH510300': '510300',
            },
            'yfinance': {
                'SPX': '^GSPC',
                'BTC': 'BTC-USD',
                'ETH': 'ETH-USD',
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

    def _is_us_stock(self, symbol: str) -> bool:
        try:
            s = safe_str(symbol).upper()
            return bool(re.match(r'^[A-Z]{1,5}$', s))
        except Exception:
            return False

    def fetch_market_data(
        self,
        symbol: str,
        interval: str,
        start_datetime: datetime,
        end_datetime: datetime,
        market_data_source: str = None
    ) -> pd.DataFrame:
        """Fetch OHLCV data using explicit data source priority from configuration."""
        start_str = start_datetime.strftime('%Y%m%d')
        end_str = end_datetime.strftime('%Y%m%d')
        start_ts = int(start_datetime.timestamp())
        end_ts = int(end_datetime.timestamp())

        preferred = safe_str(market_data_source).strip().lower() if market_data_source else self.data_fetcher.current_source
        sources = []
        if preferred in self.data_fetcher.sources:
            sources.append(preferred)
        for s in self.data_fetcher.sources:
            if s not in sources:
                sources.append(s)

        last_error = None
        for source in sources:
            try:
                if source == 'akshare':
                    df = self.data_fetcher.fetch_akshare_data(
                        self.symbol_mapping['akshare'].get(symbol, symbol),
                        self.timeframe_mapping['akshare'].get(interval, 'daily'),
                        start_str,
                        end_str
                    )
                elif source == 'yfinance':
                    df = self.data_fetcher.fetch_yfinance_data_with_datetime(
                        self.symbol_mapping['yfinance'].get(symbol, symbol),
                        self.timeframe_mapping['yfinance'].get(interval, '1d'),
                        start_datetime,
                        end_datetime
                    )
                else:
                    df = pd.DataFrame()

                if not df.empty:
                    return df
            except Exception as e:
                last_error = safe_str(e)
                continue

        if last_error:
            print(f"All data sources failed to fetch data for {symbol}: {last_error}")
        else:
            print(f"All data sources failed to fetch data for {symbol}")
        return pd.DataFrame()

    # Keep other methods unchanged, only modify data fetching part
    def run_analysis(self, df: pd.DataFrame, asset_name: str, timeframe: str, generate_charts: bool = False, trading_strategy: str = 'both', account_state: Optional[Dict[str, Any]] = None, positions: Optional[List[Dict[str, Any]]] = None, candidates: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Run the trading analysis on the provided DataFrame."""
        try:
            asset_name = safe_str(asset_name)
            timeframe = safe_str(timeframe)

            df_slice = df
            required_price_columns = ["Open", "High", "Low", "Close"]
            has_datetime_column = "Datetime" in df_slice.columns
            has_datetime_index = df_slice.index.name == "Datetime" or isinstance(df_slice.index, pd.DatetimeIndex)

            if not all(col in df_slice.columns for col in required_price_columns) or (not has_datetime_column and not has_datetime_index):
                return {
                    "success": False,
                    "error": f"Missing required columns. Available columns: {list(df_slice.columns)}, Index: {df_slice.index.name}"
                }

            df_slice_dict = {}
            if has_datetime_index:
                df_slice = df_slice.reset_index()
                has_datetime_column = True

            if has_datetime_column:
                try:
                    df_slice_dict['Datetime'] = df_slice['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                except Exception:
                    df_slice_dict['Datetime'] = [safe_str(dt) for dt in df_slice['Datetime'].tolist()]

            for col in required_price_columns:
                try:
                    df_slice_dict[col] = [float(x) if pd.notna(x) else 0.0 for x in df_slice[col].tolist()]
                except Exception:
                    df_slice_dict[col] = [safe_str(x) for x in df_slice[col].tolist()]

            print(f"\n📊 [数据转换] {asset_name} 转换为字典后:")
            if 'Close' in df_slice_dict:
                close_prices = df_slice_dict['Close']
                print(f"  收盘价数量: {len(close_prices)}")
                print(f"  收盘价范围: ${min(close_prices):.2f} - ${max(close_prices):.2f}")
                print(f"  前3条: {close_prices[:3]}")
                print(f"  后3条: {close_prices[-3:]}")

            display_timeframe = timeframe
            if timeframe.endswith('h'):
                display_timeframe += 'our'
            elif timeframe.endswith('m'):
                display_timeframe += 'in'
            elif timeframe.endswith('d'):
                display_timeframe += 'ay'

            print(f"📊 [DEBUG] run_analysis 中 generate_charts={generate_charts}")
            if generate_charts:
                print(f"✅ [DEBUG] 启用图表生成模式")
                analysis_result = self.trading_graph.analyze(
                    df_slice_dict,
                    asset_name,
                    display_timeframe,
                    trading_strategy,
                    account_state=account_state,
                    positions=positions,
                    candidates=candidates,
                )
            else:
                print(f"⚠️ [DEBUG] 仅文本分析模式，不生成图表")
                analysis_result = self.trading_graph.analyze_text_only(
                    df_slice_dict,
                    asset_name,
                    display_timeframe,
                    trading_strategy,
                    account_state=account_state,
                    positions=positions,
                    candidates=candidates,
                )

            final_state = analysis_result.get("final_state", {})
            print(f"从TradingGraph提取的final_state键: {list(final_state.keys())}")
            print(f"趋势报告长度: {len(final_state.get('trend_report', ''))}")
            print(f"指标报告长度: {len(final_state.get('indicator_report', ''))}")
            print(f"形态报告长度: {len(final_state.get('pattern_report', ''))}")

            if isinstance(final_state, dict):
                for key, value in list(final_state.items()):
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

    def extract_analysis_results(self, results: Dict[str, Any], workspace_writeback: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract and format analysis results for web display."""
        if not results["success"]:
            return {"error": safe_str(results["error"])}

        final_state = results["final_state"]

        # Extract analysis results from state fields with safe string conversion
        technical_indicators = safe_str(final_state.get("indicator_report", ""))
        pattern_analysis = safe_str(final_state.get("pattern_report", ""))
        trend_analysis = safe_str(final_state.get("trend_report", ""))
        final_decision_raw = safe_str(final_state.get("final_trade_decision", ""))
        decision_payload = final_state.get("decision_payload", {}) or {}
        single_name_score = final_state.get("single_name_score", {}) or {}

        pattern_chart = safe_str(final_state.get("pattern_image", ""))
        trend_chart = safe_str(final_state.get("trend_image", ""))
        pattern_image_filename = safe_str(final_state.get("pattern_image_filename", ""))
        trend_image_filename = safe_str(final_state.get("trend_image_filename", ""))

        if decision_payload:
            final_decision = {
                "decision": safe_str(decision_payload.get('decision', 'N/A')),
                "risk_reward_ratio": safe_str(decision_payload.get('risk_reward_ratio', 'N/A')),
                "forecast_horizon": safe_str(decision_payload.get('forecast_horizon', 'N/A')),
                "justification": safe_str(decision_payload.get('justification', 'N/A'))
            }
        elif final_decision_raw:
            final_decision = {"raw": safe_str(final_decision_raw)}
        else:
            final_decision = {}

        normalized_score = {
            "decision": safe_str(single_name_score.get("decision", "N/A")),
            "confidence": safe_str(single_name_score.get("confidence", "N/A")),
            "risk_reward_ratio": safe_str(single_name_score.get("risk_reward_ratio", "N/A")),
            "forecast_horizon": safe_str(single_name_score.get("forecast_horizon", "N/A")),
            "justification": safe_str(single_name_score.get("justification", "N/A")),
            "recommended_book": safe_str(single_name_score.get("recommended_book", "N/A")),
            "recommended_action": safe_str(single_name_score.get("recommended_action", "N/A")),
            "trend_score": single_name_score.get("trend_score", 0),
            "entry_score": single_name_score.get("entry_score", 0),
            "valuation_stretch_score": single_name_score.get("valuation_stretch_score", 0),
            "catalyst_score": single_name_score.get("catalyst_score", 0),
            "volatility_score": single_name_score.get("volatility_score", 0),
            "suggested_position_range": safe_str(single_name_score.get("suggested_position_range", "N/A")),
            "invalidation_price": safe_str(single_name_score.get("invalidation_price", "N/A")),
        }
        dashboard_payload = final_state.get("dashboard_payload", {}) or {}
        account_summary = dashboard_payload.get("account_summary", {}) or {}
        portfolio_checks = dashboard_payload.get("portfolio_checks", []) or []
        normalized_checks = [
            {
                "name": safe_str(check.get("name", "N/A")),
                "status": safe_str(check.get("status", "N/A")),
                "reason": safe_str(check.get("reason", "N/A")),
            }
            for check in portfolio_checks
            if isinstance(check, dict)
        ]
        candidate_summary = dashboard_payload.get("candidate_summary", {}) or {}

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
            "final_decision": final_decision,
            "single_name_score": normalized_score,
            "decision_payload": decision_payload,
            "account_state": final_state.get("account_state", {}),
            "positions": final_state.get("positions", []),
            "candidates": final_state.get("candidates", []),
            "portfolio_directive": final_state.get("portfolio_directive", {}),
            "dashboard_payload": {
                "account_summary": account_summary,
                "portfolio_checks": normalized_checks,
                "factor_exposure_summary": dashboard_payload.get("factor_exposure_summary", {}),
                "candidate_summary": (
                    {
                        "ticker": safe_str(candidate_summary.get("ticker", "N/A")),
                        "recommended_action": safe_str(candidate_summary.get("recommended_action", "N/A")),
                        "recommended_book": safe_str(candidate_summary.get("recommended_book", "N/A")),
                        "suggested_position_range": safe_str(candidate_summary.get("suggested_position_range", "N/A")),
                        "existing_position": bool(candidate_summary.get("existing_position", False)),
                        "candidate_factor_tags": [safe_str(tag) for tag in candidate_summary.get("candidate_factor_tags", [])],
                        "blocked_reasons": [safe_str(reason) for reason in candidate_summary.get("blocked_reasons", [])],
                    }
                    if candidate_summary else {}
                ),
                "manager_actions": [safe_str(action) for action in dashboard_payload.get("manager_actions", [])],
            },
            "workspace_writeback": workspace_writeback or {"updated": False, "message": "No workspace writeback attempted."}
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

# Initialize environment
setup_environment()

# Initialize database manager at startup
print("🔍 初始化数据库管理器...")
try:
    from services.database import get_database_manager
    db_manager = get_database_manager()
    print(f"✅ 数据库管理器初始化成功")
    print(f"📁 数据库路径: {db_manager.db_path}")
    
    # 检查数据库文件是否存在
    import os
    if os.path.exists(db_manager.db_path):
        file_size = os.path.getsize(db_manager.db_path)
        print(f"📊 数据库文件大小: {file_size} 字节")
    else:
        print("⚠️  数据库文件尚未创建，将在第一次API调用时创建")
        
except Exception as e:
    print(f"❌ 数据库管理器初始化失败: {e}")
    import traceback
    traceback.print_exc()

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

def _build_stage2_workspace_response(workspace=None, workspace_name='default', created_at=None, updated_at=None):
    response_workspace = normalize_workspace_payload(workspace or {})
    response_workspace['workspace_name'] = workspace_name or 'default'
    response_workspace['created_at'] = created_at
    response_workspace['updated_at'] = updated_at
    return response_workspace


def _latest_close_price(df):
    if df is None or df.empty or 'Close' not in df.columns:
        return None
    try:
        return float(df['Close'].iloc[-1])
    except (TypeError, ValueError, IndexError):
        return None


def _build_analysis_summary(formatted_results):
    if not isinstance(formatted_results, dict):
        return ''

    final_decision = formatted_results.get('final_decision')
    if isinstance(final_decision, dict):
        justification = safe_str(final_decision.get('justification', '')).strip()
        if justification and justification != 'N/A':
            return justification
        raw = safe_str(final_decision.get('raw', '')).strip()
        if raw:
            return raw

    for field in ('trend_analysis', 'pattern_analysis', 'technical_indicators'):
        value = safe_str(formatted_results.get(field, '')).strip()
        if value:
            return value[:500]
    return ''


def _attempt_workspace_writeback(asset, latest_price, formatted_results):
    asset_ticker = safe_str(asset).strip().upper()
    if not asset_ticker:
        return {"updated": False, "message": "Missing asset ticker for workspace writeback."}
    if latest_price is None:
        return {"updated": False, "message": f"No latest close available for {asset_ticker} workspace writeback."}

    stored_workspace = db_manager.get_stage2_workspace('default')
    if not stored_workspace:
        return {"updated": False, "message": f"No saved Stage 2 workspace found for {asset_ticker}."}

    analysis_summary = _build_analysis_summary(formatted_results)
    refreshed_workspace, metadata = refresh_position_after_analysis(
        stored_workspace,
        asset=asset_ticker,
        latest_price=latest_price,
        analysis_summary=analysis_summary,
    )

    if metadata.get('updated'):
        saved_workspace = db_manager.save_stage2_workspace(refreshed_workspace, workspace_name='default')
        metadata['workspace_name'] = saved_workspace.get('workspace_name', 'default')
        metadata['workspace_updated_at'] = saved_workspace.get('updated_at')
    else:
        metadata['workspace_name'] = stored_workspace.get('workspace_name', 'default')
    return metadata


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _extract_llm_message_content(response):
    try:
        content = response.choices[0].message.content
    except Exception:
        content = None

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get('text')
                if text:
                    parts.append(safe_str(text))
            else:
                text = getattr(item, 'text', None)
                if text:
                    parts.append(safe_str(text))
        return "\n".join(parts)
    return safe_str(content or '')


def _parse_account_analysis_payload(raw_text):
    text = safe_str(raw_text).strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

    def _load_or_raise(payload_text):
        parsed_payload = json.loads(payload_text)
        if not isinstance(parsed_payload, dict):
            raise ValueError('Account analysis response JSON must be an object.')
        return parsed_payload

    def _insert_missing_comma(payload_text, error):
        if 'Expecting' not in safe_str(error) or 'delimiter' not in safe_str(error):
            return None

        insert_at = getattr(error, 'pos', None)
        if not isinstance(insert_at, int) or insert_at < 0 or insert_at > len(payload_text):
            return None

        left = insert_at - 1
        while left >= 0 and payload_text[left].isspace():
            left -= 1

        right = insert_at
        while right < len(payload_text) and payload_text[right].isspace():
            right += 1

        if left < 0 or right >= len(payload_text):
            return None

        left_char = payload_text[left]
        right_char = payload_text[right]
        left_candidates = set(']}"0123456789eE')
        right_candidates = set('{["-0123456789tfn')

        if left_char not in left_candidates or right_char not in right_candidates:
            return None

        return payload_text[:right] + ',' + payload_text[right:]

    def _load_with_missing_comma_repairs(payload_text, max_repairs=6):
        current_text = payload_text
        seen_texts = {current_text}

        for _ in range(max_repairs + 1):
            try:
                return _load_or_raise(current_text)
            except json.JSONDecodeError as parse_error:
                repaired_text = _insert_missing_comma(current_text, parse_error)
                if repaired_text is None or repaired_text in seen_texts:
                    raise
                current_text = repaired_text
                seen_texts.add(current_text)

        return _load_or_raise(current_text)

    try:
        return _load_with_missing_comma_repairs(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start == -1 or end <= start:
            raise ValueError('Account analysis response did not contain JSON output.')

        sliced = text[start:end]
        return _load_with_missing_comma_repairs(sliced)


def _normalize_account_analysis_payload(payload):
    payload = payload if isinstance(payload, dict) else {}

    def ensure_list(value):
        return value if isinstance(value, list) else []

    def ensure_dict(value):
        return value if isinstance(value, dict) else {}

    try:
        health_score = float(payload.get('portfolio_health_score', 0) or 0)
    except (TypeError, ValueError):
        health_score = 0.0

    return {
        'summary': safe_str(payload.get('summary', '')).strip(),
        'portfolio_health_score': round(health_score, 4),
        'holding_health': ensure_list(payload.get('holding_health')),
        'pnl_breakdown': ensure_dict(payload.get('pnl_breakdown')),
        'concentration_risks': ensure_list(payload.get('concentration_risks')),
        'crowded_exposures': ensure_list(payload.get('crowded_exposures')),
        'manager_actions': ensure_list(payload.get('manager_actions')),
    }


def _format_account_analysis_markdown(workspace_name, created_at, analysis_payload):
    summary = safe_str(analysis_payload.get('summary', '')).strip() or '未提供摘要。'
    score = analysis_payload.get('portfolio_health_score', 0)

    lines = [
        '# 账户分析',
        '',
        f'- 工作区: {safe_str(workspace_name)}',
        f'- 生成时间: {safe_str(created_at)}',
        f'- 组合健康分: {score}',
        '',
        '## 摘要',
        summary,
    ]

    manager_actions = analysis_payload.get('manager_actions') or []
    lines.extend(['', '## 经理动作'])
    if manager_actions:
        for action in manager_actions:
            if isinstance(action, dict):
                label = safe_str(action.get('action') or action.get('title') or action.get('ticker') or '动作').strip()
                detail = safe_str(action.get('reason') or action.get('summary') or action.get('notes') or '').strip()
                lines.append(f'- {label}: {detail}' if detail else f'- {label}')
            else:
                lines.append(f'- {safe_str(action)}')
    else:
        lines.append('- 无')

    holding_health = analysis_payload.get('holding_health') or []
    lines.extend(['', '## 持仓健康度'])
    if holding_health:
        for item in holding_health:
            if isinstance(item, dict):
                ticker = safe_str(item.get('ticker') or item.get('symbol') or '未知').strip()
                status = safe_str(item.get('status') or item.get('health') or '').strip()
                detail = safe_str(item.get('summary') or item.get('reason') or item.get('notes') or '').strip()
                base_line = f'- {ticker}'
                if status:
                    base_line += f' [{status}]'
                if detail:
                    base_line += f': {detail}'
                lines.append(base_line)
            else:
                lines.append(f'- {safe_str(item)}')
    else:
        lines.append('- 无')

    for section_title, field_name in (
        ('集中度风险', 'concentration_risks'),
        ('拥挤暴露', 'crowded_exposures'),
    ):
        items = analysis_payload.get(field_name) or []
        lines.extend(['', f'## {section_title}'])
        if items:
            for item in items:
                if isinstance(item, dict):
                    label = safe_str(item.get('ticker') or item.get('name') or item.get('risk') or '条目').strip()
                    detail = safe_str(item.get('summary') or item.get('reason') or item.get('notes') or '').strip()
                    lines.append(f'- {label}: {detail}' if detail else f'- {label}')
                else:
                    lines.append(f'- {safe_str(item)}')
        else:
            lines.append('- 无')

    pnl_breakdown = analysis_payload.get('pnl_breakdown') or {}
    lines.extend(['', '## 盈亏拆解'])
    if pnl_breakdown:
        for key, value in pnl_breakdown.items():
            lines.append(f'- {safe_str(key)}: {safe_str(value)}')
    else:
        lines.append('- 无')

    return '\n'.join(lines)


def _refresh_workspace_prices_for_account_analysis(workspace, market_data_source=None):
    refreshed_workspace = normalize_workspace_payload(deepcopy(workspace or {}))
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=10)
    updated_positions = []
    skipped_positions = []
    refresh_timestamp = _utc_now_iso()

    for position in refreshed_workspace.get('positions', []):
        ticker = safe_str(position.get('ticker', '')).strip().upper()
        if not ticker:
            skipped_positions.append({'ticker': '', 'reason': 'Missing ticker.'})
            continue

        try:
            df = analyzer.fetch_market_data(ticker, '1d', start_dt, now, market_data_source=market_data_source)
        except Exception as e:
            skipped_positions.append({'ticker': ticker, 'reason': safe_str(e)})
            continue

        latest_price = _latest_close_price(df)
        if latest_price is None:
            skipped_positions.append({'ticker': ticker, 'reason': 'No latest close available.'})
            continue

        position['latest_price'] = float(latest_price)
        position['price_source'] = PRICE_SOURCE_ANALYSIS_REFRESH
        position['last_price_update_at'] = refresh_timestamp
        updated_positions.append({'ticker': ticker, 'latest_price': float(latest_price)})

    refreshed_workspace = recalculate_account_state(refreshed_workspace)
    metadata = {
        'updated_count': len(updated_positions),
        'updated_positions': updated_positions,
        'skipped_positions': skipped_positions,
        'refreshed_at': refresh_timestamp,
    }
    return refreshed_workspace, metadata


def _account_analysis_output_dir(workspace_name):
    safe_workspace_name = re.sub(r'[^A-Za-z0-9._-]+', '-', safe_str(workspace_name).strip() or 'default')
    return ACCOUNT_ANALYSIS_DIR / safe_workspace_name


def _list_all_account_analysis_history():
    if not ACCOUNT_ANALYSIS_DIR.exists() or not ACCOUNT_ANALYSIS_DIR.is_dir():
        return []

    history = []
    for child in ACCOUNT_ANALYSIS_DIR.iterdir():
        if child.is_dir():
            history.extend(list_account_analysis_history(child))
    history.sort(key=lambda item: str(item.get('created_at') or ''), reverse=True)
    return history[:10]


def _run_single_position_core_skill(ticker, df, workspace):
    result = analyzer.run_analysis(
        df=df,
        asset_name=ticker,
        timeframe='1d',
        generate_charts=False,
        trading_strategy='high_frequency',
        account_state=workspace.get('account_state', {}),
        positions=workspace.get('positions', []),
        candidates=workspace.get('candidates', []),
    )
    if not result.get('success'):
        raise ValueError(f"Core skill run failed for {ticker}: {safe_str(result.get('error'))}")
    return analyzer.extract_analysis_results(result)


def _run_core_skill_account_block(workspace_name, workspace, market_data_source=None):
    return build_core_skill_block(
        workspace_name=workspace_name,
        workspace=workspace,
        fetch_market_data=analyzer.fetch_market_data,
        analyze_position=_run_single_position_core_skill,
        market_data_source=market_data_source,
    )


def _run_account_analysis_llm(workspace_name, workspace):
    provider_config = analyzer.llm_provider.providers.get(analyzer.llm_provider.current_provider, {})
    model = provider_config.get('models', ['gpt-4o-mini'])[0]
    client = analyzer.llm_provider.get_client()
    prompt_payload = {
        'workspace_name': safe_str(workspace_name),
        'account_state': workspace.get('account_state', {}),
        'positions': workspace.get('positions', []),
        'candidates': workspace.get('candidates', []),
        'output_schema': {
            'summary': 'string',
            'portfolio_health_score': 'number',
            'holding_health': 'array',
            'pnl_breakdown': 'object',
            'concentration_risks': 'array',
            'crowded_exposures': 'array',
            'manager_actions': 'array',
        },
    }

    def _request_structured_repair(raw_output_text):
        repair_payload = {
            'task': 'convert_to_valid_json',
            'output_schema': prompt_payload['output_schema'],
            'raw_output': safe_str(raw_output_text),
        }
        repair_response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'system',
                    'content': '你是JSON修复器。请将输入文本修复或改写为一个合法JSON对象，且严格遵循给定schema；不要输出解释、不要输出Markdown代码块。所有文本字段必须使用简体中文。',
                },
                {
                    'role': 'user',
                    'content': json.dumps(repair_payload, ensure_ascii=False),
                },
            ],
            max_tokens=1800,
            temperature=0,
        )
        repaired_raw = _extract_llm_message_content(repair_response)
        return _parse_account_analysis_payload(repaired_raw)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': '你是一名专业的组合经理。请基于账户快照进行组合级复盘，并且仅返回符合给定 schema 的合法 JSON 对象。所有文本字段（如 summary、reason、notes、action）必须使用简体中文。重点关注风险、集中度、持仓健康度和可执行的下一步动作。',
            },
            {
                'role': 'user',
                'content': json.dumps(prompt_payload, ensure_ascii=False),
            },
        ],
        max_tokens=1800,
        temperature=0.2,
    )
    raw_content = _extract_llm_message_content(response)
    try:
        parsed_payload = _parse_account_analysis_payload(raw_content)
    except (json.JSONDecodeError, ValueError) as parse_error:
        raw_text = safe_str(raw_content).strip()
        try:
            parsed_payload = _request_structured_repair(raw_text or safe_str(parse_error))
        except (json.JSONDecodeError, ValueError, Exception):
            fallback_summary = (
                f"模型输出未返回JSON，已按文本回退。原始输出：{raw_text}"
                if raw_text
                else f"模型输出未返回JSON，解析失败：{safe_str(parse_error)}"
            )
            parsed_payload = {
                'summary': fallback_summary[:1200],
                'portfolio_health_score': 0,
                'holding_health': [],
                'pnl_breakdown': {},
                'concentration_risks': [],
                'crowded_exposures': [],
                'manager_actions': [],
            }
    return _normalize_account_analysis_payload(parsed_payload)


@app.route('/api/stage2-workspace', methods=['GET'])
def get_stage2_workspace():
    try:
        workspace_name = (request.args.get('workspace_name') or 'default').strip() or 'default'
        stored_workspace = db_manager.get_stage2_workspace(workspace_name)
        if stored_workspace:
            return jsonify({"success": True, "workspace": _build_stage2_workspace_response(
                workspace=stored_workspace,
                workspace_name=stored_workspace.get('workspace_name', workspace_name),
                created_at=stored_workspace.get('created_at'),
                updated_at=stored_workspace.get('updated_at'),
            )})

        return jsonify({"success": True, "workspace": _build_stage2_workspace_response(workspace_name=workspace_name)})
    except Exception as e:
        return jsonify({"success": False, "error": safe_str(e)}), 500


@app.route('/api/stage2-workspace', methods=['POST'])
def save_stage2_workspace():
    try:
        payload = request.get_json() or {}
        workspace_name = str(payload.get('workspace_name', 'default')).strip() or 'default'
        normalized_workspace = normalize_workspace_payload(payload)
        saved_workspace = db_manager.save_stage2_workspace(normalized_workspace, workspace_name=workspace_name)
        return jsonify({"success": True, "workspace": _build_stage2_workspace_response(
            workspace=saved_workspace,
            workspace_name=saved_workspace.get('workspace_name', workspace_name),
            created_at=saved_workspace.get('created_at'),
            updated_at=saved_workspace.get('updated_at'),
        )})
    except Exception as e:
        return jsonify({"success": False, "error": safe_str(e)}), 500


@app.route('/api/account-analysis', methods=['POST'])
def run_account_analysis():
    try:
        payload = request.get_json() or {}
        workspace_name = str(payload.get('workspace_name', 'default')).strip() or 'default'
        market_data_source = payload.get('market_data_source')

        stored_workspace = db_manager.get_stage2_workspace(workspace_name)
        if not stored_workspace:
            return jsonify({"success": False, "error": f"Workspace not found: {workspace_name}"}), 404

        refreshed_workspace, price_refresh = _refresh_workspace_prices_for_account_analysis(
            stored_workspace,
            market_data_source=market_data_source,
        )
        saved_workspace = db_manager.save_stage2_workspace(refreshed_workspace, workspace_name=workspace_name)

        analysis_payload = _run_account_analysis_llm(workspace_name, saved_workspace)
        core_skill_block = _run_core_skill_account_block(
            workspace_name=workspace_name,
            workspace=saved_workspace,
            market_data_source=market_data_source,
        )
        analysis_payload['core_skill_block'] = core_skill_block
        created_at = _utc_now_iso()
        markdown_body = _format_account_analysis_markdown(workspace_name, created_at, analysis_payload)
        artifacts = save_account_analysis_artifacts(
            output_dir=_account_analysis_output_dir(workspace_name),
            workspace_name=workspace_name,
            analysis_payload=analysis_payload,
            markdown_body=markdown_body,
            created_at=created_at,
        )

        return jsonify({
            "success": True,
            "workspace_name": workspace_name,
            "workspace": _build_stage2_workspace_response(
                workspace=saved_workspace,
                workspace_name=saved_workspace.get('workspace_name', workspace_name),
                created_at=saved_workspace.get('created_at'),
                updated_at=saved_workspace.get('updated_at'),
            ),
            "price_refresh": price_refresh,
            "analysis": analysis_payload,
            "analysis_markdown": markdown_body,
            "core_skill_block": core_skill_block,
            "artifacts": artifacts,
        })
    except Exception as e:
        return jsonify({"success": False, "error": safe_str(e)}), 500


@app.route('/api/account-analysis/history', methods=['GET'])
def get_account_analysis_history():
    try:
        workspace_name = (request.args.get('workspace_name') or '').strip()
        if workspace_name:
            history = list_account_analysis_history(_account_analysis_output_dir(workspace_name))
        else:
            history = _list_all_account_analysis_history()
        return jsonify({"success": True, "history": history})
    except Exception as e:
        return jsonify({"success": False, "error": safe_str(e)}), 500


# Keep other routes unchanged
@app.route('/')
def index():
    return render_template('demo_new.html')

@app.route('/QuantAgent')
def QuantAgent():
    return render_template('demo_new.html')

@app.route('/output')
def output():
    """Display analysis results page"""
    try:
        # Check for comparison mode
        mode = request.args.get('mode')
        if mode == 'compare':
            id1 = request.args.get('id1')
            id2 = request.args.get('id2')
            if id1 and id2:
                try:
                    print(f"📊 [DEBUG] Loading comparison results: id1={id1}, id2={id2}")
                    record1 = db_manager.get_analysis_history_by_id(int(id1))
                    record2 = db_manager.get_analysis_history_by_id(int(id2))
                    
                    results_1 = record1.get('result_details', {}) if record1 else {}
                    results_2 = record2.get('result_details', {}) if record2 else {}
                    
                    # Add cache info
                    if record1:
                        results_1['cache_info'] = {
                            'cache_id': id1,
                            'cache_timestamp': record1.get('created_at'),
                            'is_cached': True
                        }
                    if record2:
                        results_2['cache_info'] = {
                            'cache_id': id2,
                            'cache_timestamp': record2.get('created_at'),
                            'is_cached': True
                        }
                        
                    return render_template('output.html', comparison_mode=True, results_1=results_1, results_2=results_2)
                except Exception as e:
                    print(f"❌ [DEBUG] Error loading comparison results: {safe_str(e)}")
                    # Fall through to standard error handling or single result display
        
        # 优先尝试通过ID从数据库加载结果
        result_id = request.args.get('id')
        if result_id:
            try:
                print(f"📊 [DEBUG] 从数据库加载结果，ID: {result_id}")
                history_record = db_manager.get_analysis_history_by_id(int(result_id))
                if history_record:
                    results = history_record.get('result_details', {})
                    results['cache_info'] = {
                        'cache_id': result_id,
                        'cache_timestamp': history_record.get('created_at'),
                        'is_cached': True
                    }
                    print(f"✅ [DEBUG] 成功从数据库加载结果")
                    return render_template('output.html', results=results)

                error_msg = f"Analysis record not found for id {result_id}"
                print(f"⚠️ [DEBUG] {error_msg}")
                missing_result = {
                    "success": False,
                    "error": error_msg,
                    "asset_name": "Unknown",
                    "timeframe": "Unknown",
                    "data_length": 0,
                    "technical_indicators": error_msg,
                    "pattern_analysis": "",
                    "trend_analysis": "",
                    "final_decision": {
                        "decision": "HOLD",
                        "risk_reward_ratio": "N/A",
                        "forecast_horizon": "N/A",
                        "justification": error_msg,
                    },
                    "cache_info": {"is_cached": False},
                }
                return render_template('output.html', results=missing_result)
            except Exception as e:
                error_msg = f"Failed to load analysis record id {result_id}: {safe_str(e)}"
                print(f"❌ [DEBUG] {error_msg}")
                load_error_result = {
                    "success": False,
                    "error": error_msg,
                    "asset_name": "Unknown",
                    "timeframe": "Unknown",
                    "data_length": 0,
                    "technical_indicators": error_msg,
                    "pattern_analysis": "",
                    "trend_analysis": "",
                    "final_decision": {
                        "decision": "HOLD",
                        "risk_reward_ratio": "N/A",
                        "forecast_horizon": "N/A",
                        "justification": error_msg,
                    },
                    "cache_info": {"is_cached": False},
                }
                return render_template('output.html', results=load_error_result)
        
        # Get results from URL parameters (后备方案)
        results_param = request.args.get('results')
        if results_param:
            import urllib.parse
            try:
                print(f"原始URL参数: {results_param[:100]}...")
                # 先进行URL解码
                decoded_results = urllib.parse.unquote(results_param, encoding='utf-8')
                print(f"解码后结果: {decoded_results[:200]}...")  # 只显示前200个字符
                
                # 确保解码后的字符串使用UTF-8编码
                if isinstance(decoded_results, str):
                    decoded_results = decoded_results.encode('utf-8', errors='replace').decode('utf-8')
                
                # 解析JSON
                results = json.loads(decoded_results)
                print(f"JSON解析成功，technical_indicators长度: {len(results.get('technical_indicators', ''))}")
                
                # 确保所有字符串字段都是安全的
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, str):
                            results[key] = safe_str(value)
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, str):
                                    value[sub_key] = safe_str(sub_value)
                                    
                # 添加缓存标记信息
                if results.get('cached'):
                    results['cache_info'] = {
                        'cache_id': results.get('cache_id'),
                        'cache_timestamp': results.get('cache_timestamp'),
                        'is_cached': True
                    }
                else:
                    results['cache_info'] = {'is_cached': False}
                                    
            except Exception as decode_error:
                print(f"URL decode error: {safe_str(decode_error)}")
                print(f"错误类型: {type(decode_error).__name__}")
                # 如果解码失败，使用默认结果
                results = {
                    "success": False,
                    "error": f"Failed to decode results: {safe_str(decode_error)}",
                    "asset_name": "Unknown",
                    "timeframe": "Unknown",
                    "data_length": 0,
                    "cache_info": {"is_cached": False}
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
                },
                "cache_info": {"is_cached": False}
            }
        
        return render_template('output.html', results=results)
        
    except Exception as e:
        # If there's an error parsing results, show error page
        error_results = {
            "success": False,
            "error": f"Error loading results: {safe_str(e)}",
            "asset_name": "Unknown",
            "timeframe": "Unknown",
            "data_length": 0,
            "cache_info": {"is_cached": False}
        }
        return render_template('output.html', results=error_results)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        asset = data.get('asset')
        timeframe = data.get('timeframe')
        market_data_source = data.get('market_data_source')
        start_date = data.get('start_date')
        start_time = data.get('start_time', '00:00')
        end_date = data.get('end_date')
        end_time = data.get('end_time', '23:59')
        redirect_to_output = data.get('redirect_to_output', False)
        generate_charts = data.get('generate_charts', False)  # 新增参数，默认关闭图表生成
        trading_strategy = data.get('trading_strategy', 'high_frequency')  # 新增交易策略参数，默认高频交易
        session_id = data.get('session_id')  # 新增：接收前端传递的session_id
        account_state = data.get('account_state') or {}
        positions = data.get('positions') or []
        candidates = data.get('candidates') or []

        analysis_params = {"market_data_source": market_data_source or analyzer.data_fetcher.current_source}
        portfolio_cache_key = build_portfolio_cache_key(account_state, positions, candidates)
        if portfolio_cache_key:
            analysis_params["portfolio_cache_key"] = portfolio_cache_key
        
        # 添加日志打印，确认策略参数是否正确传递
        print(f"[DEBUG] 收到的交易策略参数: {trading_strategy}")
        
        # Validate required parameters
        if not asset or not timeframe or not start_date or not end_date:
            return jsonify({"error": "Missing required parameters"})
        
        # Create datetime objects
        try:
            start_dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")
        except ValueError:
            return jsonify({"error": "Invalid date or time format. Please use YYYY-MM-DD for date and HH:MM for time."})
        
        # Handle Dual Strategy Mode
        if trading_strategy == 'both':
            print(f"🔍 [Dual Mode] 检查数据库缓存...")
            print(f"   📊 查询条件: {asset} {timeframe} {start_date}~{end_date}")
            
            existing_high = db_manager.check_existing_analysis(
                asset=asset,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                start_time=start_time,
                end_time=end_time,
                trading_strategy='high_frequency',
                analysis_params=analysis_params,
                max_hours_old=24
            )
            
            existing_low = db_manager.check_existing_analysis(
                asset=asset,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                start_time=start_time,
                end_time=end_time,
                trading_strategy='low_frequency',
                analysis_params=analysis_params,
                max_hours_old=24
            )
            
            if existing_high and existing_low:
                print(f"✅ [Dual Mode] 两个策略均有缓存，跳过API调用")
                id_high = existing_high['id']
                id_low = existing_low['id']
                
                if redirect_to_output:
                    redirect_url = f"/output?id1={id_high}&id2={id_low}&mode=compare"
                    print(f"📊 [DEBUG] Dual mode使用缓存重定向: {redirect_url}")
                    return jsonify({"redirect": redirect_url})
                else:
                    results_high = existing_high.get('result_details', {})
                    results_low = existing_low.get('result_details', {})
                    results_high['cache_info'] = {'cache_id': id_high, 'cache_timestamp': existing_high.get('created_at'), 'is_cached': True}
                    results_low['cache_info'] = {'cache_id': id_low, 'cache_timestamp': existing_low.get('created_at'), 'is_cached': True}
                    return jsonify({
                        "success": True,
                        "mode": "compare",
                        "results_high": results_high,
                        "results_low": results_low
                    })
            
            df = analyzer.fetch_market_data(asset, timeframe, start_dt, end_dt, market_data_source=market_data_source)
            if df.empty:
                error_message = (
                    f"无法获取 {asset} 的真实市场数据。"
                    f"请检查: 1) 股票代码是否正确; 2) 数据源是否可用; 3) 稍后重试。"
                )
                return jsonify({"error": error_message})
            
            display_name = analyzer.asset_mapping.get(asset, asset)
            latest_price = _latest_close_price(df)

            if existing_high:
                print(f"✅ [Dual Mode] 高频策略有缓存，仅执行低频分析")
                id_high = existing_high['id']
                formatted_high = existing_high.get('result_details', {})
            else:
                print(f"📊 [Dual Mode] Running High Frequency Analysis...")
                results_high = analyzer.run_analysis(
                    df,
                    display_name,
                    timeframe,
                    generate_charts,
                    'high_frequency',
                    account_state=account_state,
                    positions=positions,
                    candidates=candidates,
                )
                formatted_high_base = analyzer.extract_analysis_results(results_high)
                formatted_high = analyzer.extract_analysis_results(
                    results_high,
                    workspace_writeback=_attempt_workspace_writeback(asset, latest_price, formatted_high_base)
                )
                id_high = db_manager.save_analysis_history(
                    asset=asset, timeframe=timeframe, start_date=start_date, end_date=end_date,
                    start_time=start_time, end_time=end_time, generate_charts=generate_charts,
                    trading_strategy='high_frequency', result_summary=f"{asset} HF Analysis",
                    analysis_params=analysis_params,
                    result_details=formatted_high, status='completed', session_id=session_id, user_ip=request.remote_addr
                )

            if existing_low:
                print(f"✅ [Dual Mode] 低频策略有缓存，仅执行高频分析")
                id_low = existing_low['id']
                formatted_low = existing_low.get('result_details', {})
            else:
                print(f"📊 [Dual Mode] Running Low Frequency Analysis...")
                results_low = analyzer.run_analysis(
                    df,
                    display_name,
                    timeframe,
                    generate_charts,
                    'low_frequency',
                    account_state=account_state,
                    positions=positions,
                    candidates=candidates,
                )
                formatted_low_base = analyzer.extract_analysis_results(results_low)
                formatted_low = analyzer.extract_analysis_results(
                    results_low,
                    workspace_writeback=_attempt_workspace_writeback(asset, latest_price, formatted_low_base)
                )
                id_low = db_manager.save_analysis_history(
                    asset=asset, timeframe=timeframe, start_date=start_date, end_date=end_date,
                    start_time=start_time, end_time=end_time, generate_charts=generate_charts,
                    trading_strategy='low_frequency', result_summary=f"{asset} LF Analysis",
                    analysis_params=analysis_params,
                    result_details=formatted_low, status='completed', session_id=session_id, user_ip=request.remote_addr
                )
            
            if redirect_to_output:
                redirect_url = f"/output?id1={id_high}&id2={id_low}&mode=compare"
                print(f"📊 [DEBUG] Dual mode redirect: {redirect_url}")
                return jsonify({"redirect": redirect_url})
            else:
                return jsonify({
                    "success": True,
                    "mode": "compare",
                    "results_high": formatted_high,
                    "results_low": formatted_low
                })

        # 首先检查数据库中是否存在相同查询条件的分析结果（24小时内）
        print(f"🔍 检查数据库缓存...")
        print(f"   📊 查询条件: {asset} {timeframe} {start_date}~{end_date} {trading_strategy}")
        existing_analysis = db_manager.check_existing_analysis(
            asset=asset,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            start_time=start_time,
            end_time=end_time,
            trading_strategy=trading_strategy,
            analysis_params=analysis_params,
            max_hours_old=24
        )
        
        if existing_analysis:
            # 如果找到缓存结果，直接返回
            print(f"✅ 使用缓存的分析结果，跳过API调用")
            
            # 从缓存结果中提取数据
            result_details = existing_analysis.get('result_details', {})
            result_summary = existing_analysis.get('result_summary', '')
            
            # 构建返回结果，保留结构化字段，避免缓存路径退回旧响应格式
            formatted_results = dict(result_details) if isinstance(result_details, dict) else {}
            formatted_results.update({
                "success": True,
                "asset_name": formatted_results.get('asset_name', asset),
                "timeframe": formatted_results.get('timeframe', timeframe),
                "data_length": formatted_results.get('data_length', 0),
                "technical_indicators": formatted_results.get('technical_indicators', ''),
                "pattern_analysis": formatted_results.get('pattern_analysis', ''),
                "trend_analysis": formatted_results.get('trend_analysis', ''),
                "pattern_chart": formatted_results.get('pattern_chart', ''),
                "trend_chart": formatted_results.get('trend_chart', ''),
                "pattern_image_filename": formatted_results.get('pattern_image_filename', ''),
                "trend_image_filename": formatted_results.get('trend_image_filename', ''),
                "final_decision": formatted_results.get('final_decision', {}),
                "workspace_writeback": formatted_results.get('workspace_writeback', {"updated": False, "message": "No workspace writeback metadata available from cached result."}),
                "cached": True,
                "cache_id": existing_analysis['id'],
                "cache_timestamp": existing_analysis['created_at'],
                "cache_info": {
                    "cache_id": existing_analysis['id'],
                    "cache_timestamp": existing_analysis['created_at'],
                    "is_cached": True,
                },
            })
            formatted_results.setdefault("single_name_score", {})
            formatted_results.setdefault("decision_payload", {})
            formatted_results.setdefault("account_state", {})
            formatted_results.setdefault("positions", [])
            formatted_results.setdefault("candidates", [])
            formatted_results.setdefault("portfolio_directive", {})
            formatted_results.setdefault("dashboard_payload", {})
            
            
            if redirect_to_output:
                # 使用数据库ID传递缓存结果，避免URL过长
                cache_id = existing_analysis.get('id')
                if cache_id:
                    redirect_url = f"/output?id={cache_id}"
                    print(f"📊 [DEBUG] 使用缓存结果重定向，ID: {cache_id}")
                    return jsonify({"redirect": redirect_url})
                else:
                    # 如果没有ID，返回错误
                    return jsonify({"error": "Cache ID not found"})
            else:
                return jsonify(formatted_results)
        
        # 如果没有找到缓存结果，继续执行原有的分析流程
        print(f"🔍 未找到缓存，开始执行新的分析...")
        print(f"📊 [DEBUG] generate_charts 参数: {generate_charts}")
        print(f"📊 [DEBUG] trading_strategy 参数: {trading_strategy}")
        
        # Use new data fetching method
        df = analyzer.fetch_market_data(asset, timeframe, start_dt, end_dt, market_data_source=market_data_source)
        if df.empty:
            error_message = (
                f"无法获取 {asset} 的真实市场数据。"
                f"请检查: 1) 股票代码是否正确; 2) 数据源是否可用; 3) 稍后重试。"
                f"注意: 系统不会生成模拟数据以避免误导决策。"
            )
            return jsonify({"error": error_message})
        
        display_name = analyzer.asset_mapping.get(asset, asset)
        latest_price = _latest_close_price(df)
        print(f"📊 [DEBUG] 调用 run_analysis, generate_charts={generate_charts}")
        results = analyzer.run_analysis(
            df,
            display_name,
            timeframe,
            generate_charts,
            trading_strategy,
            account_state=account_state,
            positions=positions,
            candidates=candidates,
        )  # 传递generate_charts、trading_strategy和组合上下文参数
        formatted_results_base = analyzer.extract_analysis_results(results)
        workspace_writeback = _attempt_workspace_writeback(asset, latest_price, formatted_results_base)
        formatted_results = analyzer.extract_analysis_results(results, workspace_writeback=workspace_writeback)
        
        # 保存分析结果到数据库
        history_id = None
        try:
            history_id = db_manager.save_analysis_history(
                asset=asset,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                start_time=start_time,
                end_time=end_time,
                generate_charts=generate_charts,
                trading_strategy=trading_strategy,
                analysis_params=analysis_params,
                result_summary=f"{asset} {timeframe} 分析结果",
                result_details=formatted_results,
                status='completed',
                session_id=session_id,  # 使用前端传递的session_id
                user_ip=request.remote_addr
            )
            print(f"✅ 分析结果已保存到数据库，ID: {history_id}")
            print(f"   📊 使用的session_id: {session_id}")
        except Exception as e:
            print(f"⚠️ 保存分析结果到数据库失败: {safe_str(e)}")
        
        if redirect_to_output:
            # 使用数据库ID传递结果，避免URL过长
            if history_id:
                redirect_url = f"/output?id={history_id}"
                print(f"📊 [DEBUG] 重定向到结果页面，使用ID: {history_id}")
                return jsonify({"redirect": redirect_url})
            else:
                # 如果保存失败，返回错误
                return jsonify({"error": "Failed to save analysis results to database"})
        else:
            return jsonify(formatted_results)
        
    except Exception as e:
        error_msg = safe_str(e)
        print(f"Analysis error: {error_msg}")
        return jsonify({"error": error_msg})

# 历史记录API端点
@app.route('/api/history/save', methods=['POST'])
def save_analysis_history():
    """保存分析历史记录"""
    try:
        data = request.get_json()
        
        # 获取数据库管理器
        from services.database import get_database_manager
        db_manager = get_database_manager()
        
        # 保存历史记录
        history_id = db_manager.save_analysis_history(
            asset=data.get('asset'),
            timeframe=data.get('timeframe'),
            start_date=data.get('start_date'),
            end_date=data.get('end_date'),
            start_time=data.get('start_time'),
            end_time=data.get('end_time'),
            use_current_time=data.get('use_current_time', False),
            generate_charts=data.get('generate_charts', False),
            trading_strategy=data.get('trading_strategy'),
            analysis_params=data.get('analysis_params'),
            result_summary=data.get('result_summary'),
            result_details=data.get('result_details'),
            status=data.get('status', 'pending'),
            error_message=data.get('error_message'),
            session_id=data.get('session_id'),
            user_ip=request.remote_addr
        )
        
        return jsonify({"success": True, "history_id": history_id})
        
    except Exception as e:
        error_msg = safe_str(e)
        print(f"保存历史记录失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500

@app.route('/api/history/update', methods=['POST'])
def update_analysis_history():
    """更新分析历史记录"""
    try:
        data = request.get_json()
        history_id = data.get('history_id')
        
        if not history_id:
            return jsonify({"success": False, "error": "Missing history_id"}), 400
        
        # 获取数据库管理器
        from services.database import get_database_manager
        db_manager = get_database_manager()
        
        # 更新历史记录
        success = db_manager.update_analysis_history(
            history_id=history_id,
            result_summary=data.get('result_summary'),
            result_details=data.get('result_details'),
            status=data.get('status'),
            error_message=data.get('error_message')
        )
        
        return jsonify({"success": success})
        
    except Exception as e:
        error_msg = safe_str(e)
        print(f"更新历史记录失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500

@app.route('/api/history/list', methods=['GET'])
def get_analysis_history():
    """获取分析历史记录列表"""
    try:
        # 获取查询参数
        limit = request.args.get('limit', 50, type=int)
        asset = request.args.get('asset')
        timeframe = request.args.get('timeframe')
        status = request.args.get('status')
        days_back = request.args.get('days_back', 30, type=int)
        
        # 获取数据库管理器
        from services.database import get_database_manager
        db_manager = get_database_manager()
        
        # 获取历史记录 - 移除session_id过滤，允许跨session查看所有记录
        history_list = db_manager.get_analysis_history_list(
            limit=limit,
            asset=asset,
            timeframe=timeframe,
            status=status,
            days_back=days_back
        )
        
        return jsonify({"success": True, "history": history_list})
        
    except Exception as e:
        error_msg = safe_str(e)
        print(f"获取历史记录失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500

@app.route('/api/history/<int:history_id>', methods=['GET'])
def get_analysis_history_by_id(history_id):
    """根据ID获取分析历史记录详情"""
    try:
        # 获取数据库管理器
        from services.database import get_database_manager
        db_manager = get_database_manager()
        
        # 获取历史记录详情
        history_record = db_manager.get_analysis_history_by_id(history_id)
        
        if history_record:
            return jsonify({"success": True, "record": history_record})
        else:
            return jsonify({"success": False, "error": "Record not found"}), 404
        
    except Exception as e:
        error_msg = safe_str(e)
        print(f"获取历史记录详情失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500

@app.route('/api/history/<int:history_id>', methods=['DELETE'])
def delete_analysis_history(history_id):
    """删除分析历史记录"""
    try:
        # 获取数据库管理器
        from services.database import get_database_manager
        db_manager = get_database_manager()
        
        # 删除历史记录
        success = db_manager.delete_analysis_history(history_id)
        
        return jsonify({"success": success})
        
    except Exception as e:
        error_msg = safe_str(e)
        print(f"删除历史记录失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_analysis_history():
    """清理分析历史记录"""
    try:
        data = request.get_json() or {}
        days_older_than = data.get('days_older_than')
        
        # 获取数据库管理器
        from services.database import get_database_manager
        db_manager = get_database_manager()
        
        # 清理历史记录
        deleted_count = db_manager.clear_analysis_history(days_older_than)
        
        return jsonify({"success": True, "deleted_count": deleted_count})
        
    except Exception as e:
        error_msg = safe_str(e)
        print(f"清理历史记录失败: {error_msg}")
        return jsonify({"success": False, "error": error_msg}), 500

# 添加缺失的静态资源和API路由
@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Serve static assets"""
    try:
        assets_dir = os.path.join(_project_root, 'assets')
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
        image_dir = os.path.join(_project_root, 'data', 'images')
        
        if image_type == 'pattern':
            # 查找最新的pattern图片
            pattern_files = [f for f in os.listdir(image_dir) if f.startswith('pattern_') and f.endswith('.png')]
            if pattern_files:
                latest_file = max(pattern_files, key=lambda x: os.path.getctime(os.path.join(image_dir, x)))
                return send_file(os.path.join(image_dir, latest_file))
            else:
                # 如果没有找到分析图片，返回占位图
                placeholder_path = os.path.join(image_dir, 'pattern_placeholder.svg')
                if os.path.exists(placeholder_path):
                    return send_file(placeholder_path)
        elif image_type == 'trend':
            # 查找最新的trend图片
            trend_files = [f for f in os.listdir(image_dir) if f.startswith('trend_') and f.endswith('.png')]
            if trend_files:
                latest_file = max(trend_files, key=lambda x: os.path.getctime(os.path.join(image_dir, x)))
                return send_file(os.path.join(image_dir, latest_file))
            else:
                # 如果没有找到分析图片，返回占位图
                placeholder_path = os.path.join(image_dir, 'trend_placeholder.svg')
                if os.path.exists(placeholder_path):
                    return send_file(placeholder_path)
        
        # 如果没有找到图片，返回404
        return jsonify({"error": f"Image not found: {image_type}"}), 404
        
    except Exception as e:
        return jsonify({"error": safe_str(e)}), 404

# Other helper routes remain unchanged
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run ElonQuantAgent Web Interface')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    args = parser.parse_args()
    
    app.run(debug=True, host=args.host, port=args.port)
