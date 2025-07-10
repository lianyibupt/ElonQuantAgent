# QuantAgents

A sophisticated multi-agent trading analysis system that combines technical indicators, pattern recognition, and trend analysis using LangChain and LangGraph. The system provides both a web interface and programmatic access for comprehensive market analysis.

## 🚀 Features

- **Multi-Agent Analysis**: Three specialized agents working together:
  - **Indicator Agent**: Computes technical indicators (MACD, RSI, Stochastic, etc.)
  - **Pattern Agent**: Identifies candlestick patterns and generates visual charts
  - **Trend Agent**: Analyzes market trends and generates trend visualizations

- **Web Interface**: Modern Flask-based web application with:
  - Real-time market data from Yahoo Finance
  - Interactive asset selection (stocks, crypto, commodities, indices)
  - Multiple timeframe analysis (1m to 1d)
  - Dynamic chart generation
  - API key management

- **Supported Assets**:
  - **Stocks**: AAPL, TSLA, QQQ
  - **Crypto**: Bitcoin (BTC)
  - **Commodities**: Gold (GC), Crude Oil (CL)
  - **Indices**: S&P 500 (SPX), Dow Jones (DJI), Nasdaq (NQ)
  - **Futures**: E-mini S&P 500 (ES)
  - **Others**: VIX, US Dollar Index (DXY)

## 🛠️ Prerequisites

- Python 3.10
- Conda (recommended) or pip
- OpenAI API key
- TA-Lib library

## 📦 Installation

### 1. Create and Activate Conda Environment

```bash
conda create -n quantagents python=3.10
conda activate quantagents
```

### 2. Verify Environment Setup

```bash
conda info --envs
```
*Look for an asterisk (*) next to `quantagents` to confirm it's active*

```bash
python --version
```
*Should display: Python 3.10.18*

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install TA-Lib

**Note**: TA-Lib can be challenging to install. Follow these steps:

#### Windows:
1. Download the appropriate wheel file from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Install the downloaded wheel:
   ```bash
   pip install TA_Lib‑0.4.24‑cp310‑cp310‑win_amd64.whl
   ```

#### macOS:
```bash
brew install ta-lib
pip install TA-Lib
```

#### Linux (Ubuntu/Debian):
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

**Alternative**: If you encounter issues, visit the [TA-Lib Python repository](https://github.com/ta-lib/ta-lib-python) for detailed installation instructions.

### 5. Set Up OpenAI API Key

Create a `.env` file in the project root:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

Or set it as an environment variable:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

## 🚀 Usage

### Start the Web Interface

```bash
python web_interface.py
```

The web application will be available at `http://localhost:5000`

### Web Interface Features

1. **Asset Selection**: Choose from available stocks, crypto, commodities, and indices
2. **Timeframe Selection**: Analyze data from 1-minute to daily intervals
3. **Date Range**: Select custom date ranges for analysis
4. **Real-time Analysis**: Get comprehensive technical analysis with visualizations
5. **API Key Management**: Update your OpenAI API key through the interface

### Programmatic Usage

```python
from trading_graph import TradingGraph
import pandas as pd

# Initialize the trading system
trading_system = TradingGraph()

# Prepare your data (OHLCV format)
data = {
    "Datetime": ["2024-01-01", "2024-01-02", ...],
    "Open": [100.0, 101.0, ...],
    "High": [102.0, 103.0, ...],
    "Low": [99.0, 100.0, ...],
    "Close": [101.0, 102.0, ...]
}

# Run analysis
initial_state = {
    "kline_data": data,
    "analysis_results": None,
    "messages": [],
    "time_frame": "1d",
    "stock_name": "AAPL"
}

results = trading_system.graph.invoke(initial_state)
```

## 📁 Project Structure

```
QuantAgents/
├── web_interface.py      # Flask web application
├── trading_graph.py      # Main orchestrator for multi-agent system
├── graph_setup.py        # LangGraph configuration
├── graph_util.py         # Technical analysis tools
├── agent_state.py        # Agent state management
├── decision_agent.py     # Decision-making agent
├── indicator_agent.py    # Technical indicator agent
├── pattern_agent.py      # Pattern recognition agent
├── trend_agent.py        # Trend analysis agent
├── default_config.py     # Configuration settings
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
│   ├── index.html
│   └── test.html
└── static/              # Static assets
```

## 🔧 Configuration

The system uses the following configuration in `default_config.py`:

```python
DEFAULT_CONFIG = {
    "analyze_LLM": "gpt-4o-mini",
    "api_key": "",
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended to provide financial advice. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🐛 Troubleshooting

### Common Issues

1. **TA-Lib Installation**: If you encounter TA-Lib installation issues, refer to the [official repository](https://github.com/ta-lib/ta-lib-python) for platform-specific instructions.

2. **OpenAI API Key**: Ensure your API key is properly set in the environment or through the web interface.

3. **Data Fetching**: The system uses Yahoo Finance for data. Some symbols might not be available or have limited historical data.

4. **Memory Issues**: For large datasets, consider reducing the analysis window or using a smaller timeframe.

### Support

If you encounter any issues, please:
1. Check the troubleshooting section above
2. Review the error messages in the console
3. Ensure all dependencies are properly installed
4. Verify your OpenAI API key is valid and has sufficient credits 