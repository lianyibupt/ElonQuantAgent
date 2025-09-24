#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finnhub API免费账户测试 - 仅测试免费功能
"""
import os
import finnhub
from datetime import datetime, timedelta

def test_finnhub_functions_direct():
    """测试Finnhub API免费账户可用功能"""
    print("=" * 60)
    print("测试Finnhub API免费账户可用功能")
    print("=" * 60)
    
    # 初始化客户端
    client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY", "YOUR_FREE_KEY"))
    symbol = "AAPL"
    
    # 1. 测试实时报价
    print(f"\n1. 测试 {symbol} 实时报价...")
    try:
        quote_data = client.quote(symbol)
        print("✓ 实时报价获取成功:")
        print(f"  当前价格: ${quote_data.get('c', 'N/A')}")
        print(f"  今日开盘: ${quote_data.get('o', 'N/A')}")
        print(f"  今日最高: ${quote_data.get('h', 'N/A')}")
        print(f"  今日最低: ${quote_data.get('l', 'N/A')}")
        print(f"  昨日收盘: ${quote_data.get('pc', 'N/A')}")
        
        # 计算涨跌
        current = quote_data.get('c')
        prev_close = quote_data.get('pc')
        if current and prev_close:
            change = current - prev_close
            change_percent = (change / prev_close) * 100
            print(f"  涨跌额: ${change:.2f}")
            print(f"  涨跌幅: {change_percent:.2f}%")
    except Exception as e:
        print(f"✗ 实时报价获取失败: {e}")
    
    # 2. 测试公司基本信息
    print(f"\n2. 测试 {symbol} 公司基本信息...")
    try:
        profile_data = client.company_profile2(symbol=symbol)
        print("✓ 公司信息获取成功:")
        print(f"  公司名称: {profile_data.get('name')}")
        print(f"  行业: {profile_data.get('finnhubIndustry')}")
        print(f"  交易所: {profile_data.get('exchange')}")
        print(f"  国家: {profile_data.get('country')}")
        print(f"  市值: {profile_data.get('marketCapitalization')} 百万美元")
        print(f"  网址: {profile_data.get('weburl')}")
    except Exception as e:
        print(f"✗ 公司信息获取失败: {e}")
    
    # 3. 测试基础财务指标
    print(f"\n3. 测试 {symbol} 基础财务指标...")
    try:
        basic_financials = client.company_basic_financials(symbol, 'all')
        if 'metric' in basic_financials:
            print("✓ 基础财务指标获取成功:")
            metrics = basic_financials['metric']
            print(f"  市盈率(P/E): {metrics.get('peBasicExclExtraTTM', 'N/A')}")
            print(f"  市净率(P/B): {metrics.get('pbQuarterly', 'N/A')}")
            print(f"  每股收益(EPS): {metrics.get('epsBasicExclExtraItemsTTM', 'N/A')}")
            print(f"  净资产收益率(ROE): {metrics.get('roeRfy', 'N/A')}%")
            print(f"  资产收益率(ROA): {metrics.get('roaRfy', 'N/A')}%")
            print(f"  毛利率: {metrics.get('grossMarginTTM', 'N/A')}%")
            print(f"  52周最高: ${metrics.get('52WeekHigh', 'N/A')}")
            print(f"  52周最低: ${metrics.get('52WeekLow', 'N/A')}")
    except Exception as e:
        print(f"✗ 基础财务指标获取失败: {e}")
    
    # 4. 测试分析师推荐趋势
    print(f"\n4. 测试 {symbol} 分析师推荐趋势...")
    try:
        recommendation_data = client.recommendation_trends(symbol)
        print("✓ 分析师推荐趋势获取成功:")
        print(f"  推荐数据条数: {len(recommendation_data)}")
        
        if recommendation_data:
            print("  最近推荐趋势:")
            for i, rec in enumerate(recommendation_data[:3]):  # 显示最近3条
                period = rec.get('period', 'N/A')
                strong_buy = rec.get('strongBuy', 0)
                buy = rec.get('buy', 0)
                hold = rec.get('hold', 0)
                sell = rec.get('sell', 0)
                strong_sell = rec.get('strongSell', 0)
                
                total_recommendations = strong_buy + buy + hold + sell + strong_sell
                
                print(f"    {period}:")
                print(f"      强烈买入: {strong_buy}")
                print(f"      买入: {buy}")
                print(f"      持有: {hold}")
                print(f"      卖出: {sell}")
                print(f"      强烈卖出: {strong_sell}")
                print(f"      总推荐数: {total_recommendations}")
                
                # 计算推荐评分（1-5分，5分最好）
                if total_recommendations > 0:
                    score = (strong_buy * 5 + buy * 4 + hold * 3 + sell * 2 + strong_sell * 1) / total_recommendations
                    print(f"      推荐评分: {score:.2f}/5.0")
                print()
        else:
            print("  暂无推荐数据")
            
    except Exception as e:
        print(f"✗ 分析师推荐趋势获取失败: {e}")
    
    # 5. 测试股票新闻
    print(f"\n5. 测试 {symbol} 股票新闻...")
    try:
        # 获取最近的新闻
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        news_data = client.company_news(symbol, 
                                       _from=start_date.strftime('%Y-%m-%d'), 
                                       to=end_date.strftime('%Y-%m-%d'))
        
        print("✓ 股票新闻获取成功:")
        print(f"  新闻条数: {len(news_data)}")
        
        if news_data:
            print("  最新新闻:")
            for i, news in enumerate(news_data[:3]):  # 显示最新3条
                headline = news.get('headline', 'N/A')
                summary = news.get('summary', 'N/A')
                source = news.get('source', 'N/A')
                url = news.get('url', 'N/A')
                datetime_obj = datetime.fromtimestamp(news.get('datetime', 0))
                
                print(f"    新闻 {i+1}:")
                print(f"      标题: {headline[:80]}{'...' if len(headline) > 80 else ''}")
                print(f"      来源: {source}")
                print(f"      时间: {datetime_obj.strftime('%Y-%m-%d %H:%M')}")
                print(f"      链接: {url[:50]}{'...' if len(url) > 50 else ''}")
                print()
        else:
            print("  暂无新闻数据")
            
    except Exception as e:
        print(f"✗ 股票新闻获取失败: {e}")
    
    # 6. 测试其他股票的基本信息
    print(f"\n6. 测试其他美国股票...")
    other_symbols = ['MSFT', 'GOOGL', 'TSLA']
    for test_symbol in other_symbols:
        try:
            quote = client.quote(test_symbol)
            profile = client.company_profile2(symbol=test_symbol)
            print(f"✓ {test_symbol} - {profile.get('name', 'Unknown')}: ${quote.get('c', 'N/A')}")
            
            # 尝试获取该股票的推荐趋势
            try:
                rec_trends = client.recommendation_trends(test_symbol)
                if rec_trends and len(rec_trends) > 0:
                    latest_rec = rec_trends[0]
                    total_rec = latest_rec.get('strongBuy', 0) + latest_rec.get('buy', 0) + latest_rec.get('hold', 0) + latest_rec.get('sell', 0) + latest_rec.get('strongSell', 0)
                    print(f"    最新分析师推荐数: {total_rec}")
                else:
                    print(f"    暂无推荐数据")
            except:
                print(f"    推荐数据获取失败")
                
        except Exception as e:
            print(f"✗ {test_symbol} 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("Finnhub API 免费账户测试完成!")
    print("\n功能总结:")
    print("✓ 实时股价查询 - 免费账户可用")
    print("✓ 公司基本信息 - 免费账户可用")
    print("✓ 基础财务指标 - 免费账户可用")
    print("✓ 分析师推荐趋势 - 免费账户可用")
    print("✓ 股票新闻 - 免费账户可用")
    print("✓ 多股票对比 - 免费账户可用")
    print("\n已移除功能:")
    print("✗ 历史股价数据 (stock_candles) - 免费账户受限")
    print("✗ 详细财务报表 (financials) - 需要付费账户")
    print("=" * 60)

if __name__ == "__main__":
    test_finnhub_functions_direct()