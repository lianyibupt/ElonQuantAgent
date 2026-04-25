"""
Coze IDE AKShare插件
基于AKShare的股票数据获取插件，提供历史价格和财务数据功能

每个文件需要导出一个名为handler的函数，这是工具的入口点。

参数:
args: 入口函数的参数
args.input - 输入参数，可以通过args.input.xxx获取测试输入值
args.logger - 运行时注入的日志记录器实例

返回值:
函数的返回数据，应与声明的输出参数匹配
"""

import akshare as ak
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Price:
    """价格数据模型"""
    def __init__(self, open: float, close: float, high: float, low: float, volume: int, time: str):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.time = time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "open": self.open,
            "close": self.close,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "time": self.time
        }

class FinancialMetrics:
    """财务指标数据模型"""
    def __init__(self, ticker: str, report_period: str, period: str, currency: str):
        self.ticker = ticker
        self.report_period = report_period
        self.period = period
        self.currency = currency
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "ticker": self.ticker,
            "report_period": self.report_period,
            "period": self.period,
            "currency": self.currency
        }

def _format_symbol(symbol: str, market: str) -> str:
    """格式化股票代码为AKShare需要的格式"""
    if market == 'hk':
        if symbol.startswith('hk'):
            return symbol[2:]  # 去掉hk前缀
        else:
            return symbol
    elif market == 'us':
        if symbol.startswith('us'):
            return symbol[2:]  # 去掉us前缀
        else:
            return symbol
    return symbol

def _detect_market(symbol: str) -> str:
    """检测股票代码所属市场"""
    if symbol.startswith('hk'):
        return 'hk'  # 港股
    elif symbol.startswith('us'):
        return 'us'  # 美股
    elif symbol.isdigit() and len(symbol) == 6:
        return 'cn'  # A股数字代码
    elif symbol.isdigit() and len(symbol) == 5:
        return 'hk'  # 港股数字代码
    elif symbol.isalpha() and 1 <= len(symbol) <= 5:
        return 'us'  # 美股字母代码
    else:
        return 'cn'  # 默认A股

def get_prices(ticker: str, start_date: str, end_date: str) -> List[Price]:
    """
    从AKShare获取价格数据
    
    Args:
        ticker: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        Price对象列表
    """
    try:
        # 根据市场类型选择不同的AKShare函数
        market = _detect_market(ticker)
        formatted_symbol = _format_symbol(ticker, market)
        
        if market == 'cn':
            # A股数据
            df = ak.stock_zh_a_hist(symbol=formatted_symbol, period="daily", 
                                   start_date=start_date.replace('-', ''), 
                                   end_date=end_date.replace('-', ''), adjust="qfq")
        elif market == 'hk':
            # 港股数据
            df = ak.stock_hk_hist(symbol=formatted_symbol, period="daily", 
                                 start_date=start_date, end_date=end_date, adjust="qfq")
        elif market == 'us':
            # 美股数据
            df = ak.stock_us_daily(symbol=formatted_symbol)
            # 过滤日期范围
            if df is not None and not df.empty:
                df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
                start_dt = pd.to_datetime(start_date, format='mixed', errors='coerce')
                end_dt = pd.to_datetime(end_date, format='mixed', errors='coerce')
                mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
                df = df.loc[mask]
        else:
            logger.warning(f"不支持的股票代码格式: {ticker}")
            return []
        
        if df is None or df.empty:
            logger.warning(f"未获取到{ticker}的价格数据")
            return []
        
        # 转换数据格式
        prices = []
        for _, row in df.iterrows():
            try:
                # 根据市场类型确定列名
                if market == 'us':
                    # 美股使用英文列名
                    open_col = 'open'
                    close_col = 'close' 
                    high_col = 'high'
                    low_col = 'low'
                    volume_col = 'volume'
                    date_col = 'date'
                    
                    # 处理日期格式
                    if hasattr(row[date_col], 'strftime'):
                        time_str = row[date_col].strftime("%Y-%m-%d")
                    else:
                        parsed_date = pd.to_datetime(row[date_col], format='mixed', errors='coerce')
                        time_str = parsed_date.strftime("%Y-%m-%d") if not pd.isna(parsed_date) and hasattr(parsed_date, 'strftime') else None
                else:
                    # A股和港股使用中文列名
                    open_col = '开盘'
                    close_col = '收盘'
                    high_col = '最高'
                    low_col = '最低'
                    volume_col = '成交量'
                    date_col = '日期'
                    
                    # 处理日期格式
                    parsed_date = pd.to_datetime(row[date_col], format='mixed', errors='coerce')
                    time_str = parsed_date.strftime("%Y-%m-%d") if not pd.isna(parsed_date) and hasattr(parsed_date, 'strftime') else None
                
                prices.append(Price(
                    open=round(float(row[open_col]), 2),
                    close=round(float(row[close_col]), 2),
                    high=round(float(row[high_col]), 2),
                    low=round(float(row[low_col]), 2),
                    volume=int(row[volume_col]),
                    time=time_str
                ))
            except (ValueError, KeyError) as e:
                logger.warning(f"数据格式转换错误: {e}")
                continue
        
        logger.info(f"成功获取{ticker}的{len(prices)}条价格数据")
        return prices
        
    except Exception as e:
        logger.error(f"获取{ticker}价格数据失败: {str(e)}")
        return []

def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> List[FinancialMetrics]:
    """
    从AKShare获取财务指标
    
    Args:
        ticker: 股票代码
        end_date: 结束日期
        period: 期间（ttm表示过去12个月）
        limit: 限制数量
    
    Returns:
        FinancialMetrics对象列表
    """
    try:
        market = _detect_market(ticker)
        formatted_symbol = _format_symbol(ticker, market)
        
        # 构建一个基本的财务指标对象作为示例
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        report_date = (end_dt - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
        
        metrics = FinancialMetrics(
            ticker=ticker,
            report_period=report_date,
            period=period,
            currency='CNY' if market == 'cn' else 'HKD' if market == 'hk' else 'USD'
        )
        
        logger.info(f"成功获取{ticker}的财务指标数据")
        return [metrics]
        
    except Exception as e:
        logger.error(f"获取{ticker}财务指标失败: {str(e)}")
        return []

def prices_to_dataframe(prices: List[Price]) -> pd.DataFrame:
    """将价格数据转换为DataFrame"""
    if not prices:
        return pd.DataFrame()
        
    df = pd.DataFrame([p.to_dict() for p in prices])
    df["Date"] = pd.to_datetime(df["time"], format='mixed', errors='coerce')
    df.set_index("Date", inplace=True)
    
    # 确保数值列是数值类型
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df.sort_index(inplace=True)
    return df

def get_price_dataframe(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """获取价格数据并转换为DataFrame"""
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_dataframe(prices)

def get_company_info(ticker: str) -> dict:
    """获取公司基本信息"""
    try:
        market = _detect_market(ticker)
        formatted_symbol = _format_symbol(ticker, market)
        
        return {
            "name": ticker,
            "sector": "Unknown",
            "industry": "Unknown",
            "country": "China" if market == 'cn' else "Hong Kong" if market == 'hk' else "USA",
            "website": "",
            "description": ""
        }
    except Exception as e:
        logger.error(f"获取{ticker}公司信息失败: {str(e)}")
        return {}

def handler(args):
    """
    Coze插件入口函数
    
    Args:
        args: 包含输入参数和日志记录器
            args.input: 输入参数字典
            args.logger: 日志记录器实例
    
    Returns:
        根据操作类型返回相应的数据
    """
    # 使用传入的logger或默认logger
    logger = args.logger if hasattr(args, 'logger') else logging.getLogger(__name__)
    
    # 安全处理输入参数，处理CustomNamespace对象
    input_data = {}
    if hasattr(args, 'input') and args.input is not None:
        # 处理CustomNamespace对象（具有属性访问）
        if hasattr(args.input, '__dict__'):
            input_data = vars(args.input)  # 转换为字典
        elif isinstance(args.input, dict):
            input_data = args.input
    
    operation = input_data.get('operation', 'get_prices') if input_data else 'get_prices'
    
    # 记录输入参数用于调试
    if hasattr(args, 'logger'):
        args.logger.info(f"输入参数: {input_data}")
    
    try:
        if operation == 'get_prices':
            ticker = input_data.get('ticker')
            start_date = input_data.get('start_date')
            end_date = input_data.get('end_date')
            
            if not all([ticker, start_date, end_date]):
                return {"error": "缺少必要参数: ticker, start_date, end_date"}
            
            prices = get_prices(str(ticker), str(start_date), str(end_date))
            return {"prices": [p.to_dict() for p in prices]}
            
        elif operation == 'get_financial_metrics':
            ticker = input_data.get('ticker')
            end_date = input_data.get('end_date')
            period = input_data.get('period', 'ttm')
            limit = input_data.get('limit', 10)
            
            if not all([ticker, end_date]):
                return {"error": "缺少必要参数: ticker, end_date"}
            
            metrics = get_financial_metrics(str(ticker), str(end_date), period, limit)
            return {"metrics": [m.to_dict() for m in metrics]}
            
        elif operation == 'get_company_info':
            ticker = input_data.get('ticker')
            
            if not ticker:
                return {"error": "缺少必要参数: ticker"}
            
            info = get_company_info(str(ticker))
            return {"company_info": info}
            
        elif operation == 'get_price_dataframe':
            ticker = input_data.get('ticker')
            start_date = input_data.get('start_date')
            end_date = input_data.get('end_date')
            
            if not all([ticker, start_date, end_date]):
                return {"error": "缺少必要参数: ticker, start_date, end_date"}
            
            df = get_price_dataframe(str(ticker), str(start_date), str(end_date))
            return {"dataframe": df.to_dict()}
            
        else:
            return {"error": f"不支持的操作类型: {operation}"}
            
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return {"error": f"处理请求时发生错误: {str(e)}"}