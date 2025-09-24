# ElonQuantAgent - 多厂商API量化交易分析系统

一个支持多厂商API和数据源的量化交易分析系统，集成了OpenAI、DeepSeek等LLM提供商和akshare、finnhub等数据源。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

至少设置一个LLM API密钥：

```bash
# OpenAI (推荐)
export OPENAI_API_KEY="你的-openai-api-key"

# 或 DeepSeek
export DEEPSEEK_API_KEY="你的-deepseek-api-key"

# 可选: Finnhub数据源
export FINNHUB_API_KEY="你的-finnhub-api-key"
```

### 3. 运行项目

```bash
python run.py
```

或者直接运行Web界面：

```bash
python web_interface_new.py
```

访问 http://127.0.0.1:5000 使用系统。

## 📊 需要准备的数据

### 1. API密钥
- **必须**: 至少一个LLM API密钥（OpenAI或DeepSeek）
- **可选**: Finnhub API密钥（用于备用数据源）

### 2. 市场数据源
系统支持多种数据源，按优先级尝试：

1. **akshare** (主要) - 免费，支持：
   - A股 (000001, 399001, SH000300等)
   - 港股
   - 美股 (AAPL, TSLA等)
   - 加密货币 (BTC-USD等)

2. **finnhub** (备用) - 需要API密钥，支持：
   - 全球股票
   - 加密货币
   - 外汇
   - 指数

3. **yfinance** (兼容) - 原有支持

### 3. 内置基准数据
项目包含以下基准数据的CSV文件：
- `benchmark/btc/` - 比特币数据
- `benchmark/spx/` - S&P 500数据  
- `benchmark/dji/` - 道琼斯指数
- `benchmark/nq/` - 纳斯达克指数
- `benchmark/qqq/` - QQQ ETF
- `benchmark/vix/` - 波动率指数
- `benchmark/cl/` - 原油期货
- `benchmark/es/` - E-mini S&P期货

## 🔧 配置说明

### 环境变量
```bash
# LLM提供商
OPENAI_API_KEY="sk-..."          # OpenAI API密钥
DEEPSEEK_API_KEY="sk-..."        # DeepSeek API密钥

# 数据源
FINNHUB_API_KEY="你的finnhub密钥"  # Finnhub API密钥

# 其他配置
LLM_PROVIDER="deepseek"          # 默认LLM提供商 (openai/deepseek)
DATA_SOURCE="akshare"            # 默认数据源 (akshare/finnhub/yfinance)
```

### 支持的资产类型
- **指数**: SPX, DJI, NQ, VIX, QQQ
- **加密货币**: BTC
- **大宗商品**: CL (原油), GC (黄金)
- **A股**: 000001 (上证), 399001 (深证), SH000300 (沪深300)
- **美股**: AAPL, TSLA, 等

## 🎯 功能特性

### 多厂商LLM支持
- ✅ OpenAI GPT-4o, GPT-4o-mini
- ✅ DeepSeek deepseek-chat, deepseek-coder  
- 🔄 更多厂商支持中...

### 多数据源支持
- ✅ akshare (主要，免费)
- ✅ finnhub (备用，需要API密钥)
- ✅ yfinance (兼容)

### 交易分析功能
- 📈 技术指标分析 (MACD, RSI, Stochastic等)
- 🎯 趋势线识别
- 🕵️ 形态模式识别
- 🤖 AI驱动的交易决策

## 🐛 故障排除

### 常见问题

1. **API密钥错误**
   - 检查环境变量是否正确设置
   - 确保API密钥有足够余额

2. **数据获取失败**
   - akshare可能因网络问题失败
   - 可设置FINNHUB_API_KEY作为备用

3. **依赖安装失败**
   - 确保Python版本 >= 3.8
   - 尝试: `pip install --upgrade pip`

### 数据源优先级
1. 首先尝试akshare（免费）
2. 如果akshare失败，尝试finnhub（需要API密钥）
3. 最后尝试yfinance（兼容模式）

## 📝 开发说明

### 项目结构
```
ElonQuantAgent/
├── web_interface_new.py    # 主Web界面（多厂商支持）
├── trading_graph_new.py    # 交易图引擎（多LLM支持）
├── trend_agent.py          # 趋势分析Agent
├── decision_agent.py       # 决策Agent
├── graph_util.py           # 技术工具
├── requirements.txt        # 依赖列表
├── run.py                  # 启动脚本
└── benchmark/              # 基准数据
```

### 添加新的LLM提供商
1. 在 `MultiProviderLLM` 类中添加提供商配置
2. 实现对应的API客户端
3. 更新环境变量支持

### 添加新的数据源
1. 在 `MultiSourceDataFetcher` 类中添加数据源方法
2. 更新符号映射和时间帧转换
3. 添加对应的依赖到requirements.txt

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📞 支持

如有问题，请提交GitHub Issue或联系开发团队。