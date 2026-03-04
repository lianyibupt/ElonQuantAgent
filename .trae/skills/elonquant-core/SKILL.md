---
name: "elonquant-core"
description: "运行ElonQuant核心多Agent分析并输出Markdown报告。用户需要生成交易分析报告、复用现有Agent提示词与指标计算时调用。"
---

# ElonQuant Core Skill

## 用途
基于现有多Agent逻辑执行完整分析流程，使用akshare获取行情数据并输出Markdown报告文件。

## 适用场景
- 需要快速生成单标的交易分析报告
- 希望复用现有Agent提示词与指标计算逻辑
- 不需要前端页面、只需后端脚本输出

## 输入参数
- `symbol`: 标的代码，例如 `AAPL`
- `start`: 开始日期，格式 `YYYY-MM-DD`
- `end`: 结束日期，格式 `YYYY-MM-DD`
- `interval`: 周期，默认 `1d`
- `output`: 输出Markdown路径，默认 `analysis_output.md`
- `strategy`: 交易策略标签，默认 `high_frequency`
- `offline`: 离线模式（不调用大模型），需要时加 `--offline`
- `demo`: 演示数据（yfinance限流/无数据时使用），需要时加 `--demo`

## 调用方式
```bash
python SKILL.py --symbol AAPL --start 2024-01-01 --end 2024-02-01 --interval 1d --output report.md
```

## 输出
- Markdown 报告文件（包含指标分析、形态分析、趋势分析与最终决策）
