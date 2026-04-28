# Stage 2 账户追踪与账户分析增强设计

日期：2026-04-28  
项目：ElonQuantAgent  
范围：在现有 Stage 2 Web 工作区基础上，补齐持仓追踪、单票分析写回、账户级 AI 分析与结果留痕能力。

## 1. 目标

当前 Stage 2 已经能把 `account_state / positions / candidates` 注入分析流程，但它仍然更像“手工上下文输入器”，还不是一个可持续维护的账户工作台。

本次增强的目标是把 Stage 2 升级为一个轻量但可持续使用的账户层：

- 持续维护每只持仓的真实基础信息
- 在单票分析后自动更新该标的的最新盈亏状态
- 自动刷新账户聚合指标，辅助账户维度决策
- 提供独立的账户级 AI 分析入口，而不是把它混进单票分析
- 保留最近 10 次账户分析结果，便于回看与追溯

本次增强仍然坚持一个边界：**它是现有交易分析系统上的账户层扩展，不是一个全新的重型投资组合管理系统。**

## 2. 设计原则

### 2.1 手工维护真实输入，系统推导衍生字段

用户已经确认以下字段应由人工维护，作为真实输入：

- 单持仓：`cost_basis`、`shares`
- 账户层：`nav`、`cash`

系统自动推导或刷新以下字段：

- `latest_price`
- `market_value`
- `position_weight`
- `unrealized_pnl`
- `unrealized_pnl_pct`
- `last_price_update_at`
- 账户级总持仓市值、现金占比、最大持仓、集中度等聚合指标

这条边界很重要，因为它决定了 Stage 2 不再要求用户手工维护会频繁变化的结果字段，减少重复录入和数据漂移。

### 2.2 单票分析与账户分析分离

- 单票分析继续围绕一个 ticker 的技术面 / 结构化评分 / 交易建议展开
- 账户分析是单独入口、单独工作流、单独输出

这样可以保证：

- 单票分析保持响应快、交互轻
- 账户分析可围绕全账户上下文做更完整推理
- 不会因为每次查单票都触发整账户工作流而增加成本与噪音

### 2.3 先增强现有架构，不推翻重建

推荐继续沿用当前路线：

- Web 端仍以 `demo_new.html` 为主入口
- 后端仍由 `web_interface_new.py` 提供 API
- SQLite 继续作为本地结构化存储
- 文件夹继续承担“最近 10 次分析结果”的追溯留痕

重点是把 Stage 2 从“可试用”做成“可持续使用”，而不是一次性做成完整 PMS。

## 3. 范围

### 3.1 本次纳入范围

1. 持仓模型增强
2. 单票分析后写回最新价格与盈亏
3. 账户级聚合指标自动刷新
4. 独立的账户分析 API / Prompt / 工作流
5. 最近 10 次账户分析结果文件化留存
6. Web UI 中增加账户分析入口与历史查看入口
7. 清理当前 Stage 2 前端里重复和陈旧的 JS 逻辑，避免继续在脆弱页面上叠加功能

### 3.2 暂不纳入范围

1. 自动同步券商账户
2. 实时流式行情订阅
3. 多账户、多币种、多市场统一账本
4. 完整订单执行、成交回报、税务跟踪
5. 复杂收益归因或基准归因系统

## 4. 数据模型设计

### 4.1 账户工作区结构

现有 `stage2_workspaces` 仍可继续作为主存储容器，但 `positions` 中的对象结构需要增强。

建议工作区仍以一个默认工作区为主：

```json
{
  "workspace_name": "default",
  "account_state": {
    "nav": 0,
    "cash": 0,
    "total_market_value": 0,
    "cash_pct": 0,
    "gross_exposure": 0,
    "largest_position": "",
    "largest_position_weight": 0,
    "position_count": 0,
    "current_drawdown": 0,
    "last_aggregated_at": ""
  },
  "positions": [],
  "candidates": [],
  "notes": ""
}
```

### 4.2 持仓对象

每条持仓建议包含以下字段：

#### 手工维护字段

- `ticker`
- `book_type`：核心仓 / 战术仓 / 观察
- `cost_basis`
- `shares`
- `factor_tags`
- `tracking_status`：持续跟踪 / 暂停跟踪
- `notes`

#### 系统维护字段

- `latest_price`
- `market_value`
- `position_weight`
- `unrealized_pnl`
- `unrealized_pnl_pct`
- `price_source`
- `last_price_update_at`
- `last_analyzed_at`
- `last_analysis_summary`

建议结构示例：

```json
{
  "ticker": "MSTR",
  "book_type": "核心仓",
  "cost_basis": 152.4,
  "shares": 120,
  "factor_tags": ["btc-beta", "proxy", "momentum"],
  "tracking_status": "持续跟踪",
  "notes": "核心代理仓位",
  "latest_price": 181.2,
  "market_value": 21744,
  "position_weight": 12.4,
  "unrealized_pnl": 3456,
  "unrealized_pnl_pct": 18.89,
  "price_source": "yfinance",
  "last_price_update_at": "2026-04-28T15:10:00",
  "last_analyzed_at": "2026-04-28T15:10:10",
  "last_analysis_summary": "趋势仍强，但短线拉伸较高"
}
```

### 4.3 候选标的对象

候选标的保持轻量：

- `ticker`
- `factor_tags`
- `notes`
- `last_screened_at`

候选区的职责不是持仓记账，而是把待观察或待加仓标的放入账户视角分析上下文中。

## 5. 字段边界与计算规则

### 5.1 人工真值字段

以下字段视为人工真值，不被自动刷新覆盖：

- `cost_basis`
- `shares`
- `nav`
- `cash`
- `book_type`
- `factor_tags`
- `tracking_status`

### 5.2 系统推导字段

若持仓具备 `cost_basis + shares + latest_price`，系统可推导：

- `market_value = latest_price * shares`
- `unrealized_pnl = (latest_price - cost_basis) * shares`
- `unrealized_pnl_pct = (latest_price - cost_basis) / cost_basis * 100`

若账户具备 `nav + cash + positions.market_value`，系统可推导：

- `total_market_value = sum(position.market_value)`
- `cash_pct = cash / nav * 100`
- `gross_exposure = total_market_value / nav * 100`
- `position_weight = position.market_value / nav * 100`
- `largest_position` 与 `largest_position_weight`
- `position_count`
- `core_exposure` 与 `tactical_exposure`

### 5.3 异常处理原则

- `cost_basis <= 0` 时，不计算 `unrealized_pnl_pct`
- `shares <= 0` 的持仓保留记录，但标记为非有效持仓，不参与正常暴露计算
- `nav <= 0` 时，不计算权重与账户级比例类指标
- 若行情刷新失败，保留上次系统字段，不清空已存在结果

## 6. 单票分析后的自动写回机制

### 6.1 触发时机

当用户对单只股票执行现有分析流程后：

1. 正常完成单票分析
2. 判断该 ticker 是否存在于当前 Stage 2 工作区的持仓中
3. 如果存在，则执行写回刷新
4. 如果不存在，则不改动持仓，只返回分析结果

### 6.2 写回内容

如果命中持仓，系统更新：

- `latest_price`
- `market_value`
- `unrealized_pnl`
- `unrealized_pnl_pct`
- `position_weight`
- `last_price_update_at`
- `last_analyzed_at`
- `last_analysis_summary`

随后刷新账户聚合字段：

- `total_market_value`
- `cash_pct`
- `gross_exposure`
- `core_exposure`
- `tactical_exposure`
- `largest_position`
- `largest_position_weight`
- `position_count`
- `last_aggregated_at`

### 6.3 用户反馈

前端需要明确告诉用户写回是否发生，例如：

- “已更新 MSTR 的最新价格与浮盈亏”
- “当前标的不在 Stage 2 持仓中，未写回账户”
- “分析成功，但行情刷新失败，保留上次账户快照”

### 6.4 边界说明

该写回流程**不会**：

- 自动修改 `cost_basis`
- 自动修改 `shares`
- 自动生成买卖成交
- 自动触发账户级 AI 分析

## 7. 独立的 Account Analysis 工作流

### 7.1 触发方式

账户分析采用手动触发，作为独立入口：

- Web UI 中新增 `Account Analysis` 按钮
- 触发单独 API，例如 `POST /api/account-analysis`
- 使用当前工作区快照作为输入

### 7.2 输入结构

账户分析输入至少包括：

1. `account_state`
2. `positions`
3. `candidates`
4. 已推导的持仓与账户聚合指标
5. 可选的最近单票分析摘要

### 7.3 期望输出

用户已明确优先级，输出重点应为：

1. 持仓健康度
2. 盈亏拆解
3. 集中度 / 拥挤度
4. 组合级操作建议

建议结构化输出：

```json
{
  "portfolio_health_score": 0,
  "holding_health": [],
  "pnl_breakdown": {},
  "concentration_risks": [],
  "crowded_exposures": [],
  "manager_actions": [],
  "summary": ""
}
```

### 7.4 工作流职责

独立账户分析工作流负责：

- 拉取每只股票的最新价格或复用最近已刷新的价格
- 基于持仓推导最新 PnL / 权重 / 暴露信息
- 识别高集中、同主题过度暴露、弱趋势但高权重等问题
- 结合现有系统的单票分析框架生成账户经理视角建议

### 7.5 与单票流程关系

账户分析可以引用已有单票摘要，但不应直接依赖用户刚刚必须跑过单票分析。它应能独立运行。

## 8. 存储与追溯设计

### 8.1 结构化状态存储

继续使用 SQLite 保存当前最新工作区状态，因为它适合：

- 页面刷新后恢复
- 单票分析后的自动写回
- 前端反复编辑与保存

### 8.2 分析结果留痕

账户分析结果采用文件夹存储，原因是：

- 更适合保留完整 JSON / Markdown 输出
- 便于手工回看、对比和追溯
- 不必把富文本或大块 AI 结果塞进 SQLite

建议目录：

```text
artifacts/account_analysis/
  2026-04-28T15-20-01_default.json
  2026-04-28T15-20-01_default.md
```

### 8.3 保留策略

- 每次账户分析写入一组结果文件
- 只保留最近 10 次
- 新结果写入后删除更旧的历史文件
- 同时可以在 SQLite 或索引文件中保留一个轻量级列表，用于前端展示最近历史

### 8.4 历史查看

前端可先做最小能力：

- 显示最近 10 次账户分析时间
- 点击查看最近一次 JSON / Markdown 结果

不需要一开始就做完整 diff 或版本比较器。

## 9. Web UI 设计

### 9.1 Stage 2 工作区增强

在现有 Stage 2 tab 上补齐持仓表格列：

- ticker
- book_type
- cost_basis
- shares
- tracking_status
- factor_tags
- latest_price（只读）
- market_value（只读）
- position_weight（只读）
- unrealized_pnl（只读）
- unrealized_pnl_pct（只读）
- last_price_update_at（只读）

### 9.2 账户概览区

新增账户汇总卡片：

- NAV
- 现金
- 总持仓市值
- 现金占比
- 总暴露
- 核心仓暴露
- 战术仓暴露
- 最大持仓
- 当前回撤
- 持仓数量

### 9.3 分析入口

在 Analysis tab 中保留现有单票分析入口，同时增加：

- 单票分析后显示是否写回持仓
- `Run Account Analysis` 独立按钮
- 最近一次账户分析时间与状态提示

### 9.4 历史入口

在 Stage 2 区域增加一个最近分析历史列表：

- 时间
- 工作区名
- 结果摘要
- 查看按钮

### 9.5 前端清理要求

当前 `demo_new.html` 已经存在重复 `runAnalysis()`、重复 helper 与陈旧初始化逻辑。继续叠加 Stage 2 功能前，需要先做一次最小清理：

- 删除已失效的重复函数定义
- 收敛 Stage 2 相关状态与渲染逻辑
- 避免多个相同函数名互相覆盖
- 保持 Analysis / Stage 2 / Account Analysis 的入口逻辑清晰

这一步不是为了“重构而重构”，而是为了避免新增账户功能时继续踩到 DOM 初始化与函数覆盖问题。

## 10. 后端改造方向

### 10.1 `services/database.py`

职责：

- 扩展 Stage 2 workspace 的 positions 数据结构
- 提供读写当前工作区的方法
- 提供账户分析历史索引方法（如果需要）

建议：

- 保持 `stage2_workspaces` 主表不拆散
- 仍以 JSON-in-TEXT 保存增强后的 `positions / candidates / account_state`
- 如需历史列表展示，可增加一个轻量索引表，但不是必选项

### 10.2 `web/web_interface_new.py`

职责：

- 校验和标准化增强后的 workspace payload
- 在 `/api/analyze` 成功后对命中持仓执行写回
- 提供账户分析 API
- 提供账户分析历史读取 API

建议新增能力：

- `normalize_stage2_workspace_payload()` 扩展字段校验
- `refresh_position_from_analysis(...)`
- `recalculate_account_state(...)`
- `POST /api/account-analysis`
- `GET /api/account-analysis/history`
- `GET /api/account-analysis/latest`

### 10.3 分析层

建议新增独立的账户分析 prompt / workflow，而不是污染当前单票 `TradingGraph` 主流程。

单票流程负责：

- 单标的分析
- 命中持仓时的局部写回

账户分析流程负责：

- 遍历当前账户持仓
- 汇总账户级风险与建议
- 生成可留痕输出

## 11. 分阶段交付建议

### Phase 1：账户追踪闭环

目标：先把“持仓维护 + 单票刷新写回 + 聚合统计”跑通。

包含：

- Stage 2 持仓字段增强
- 衍生字段计算
- 单票分析后写回
- 账户汇总卡片
- 前端重复 JS 最小清理

价值：最高。因为它直接把 Stage 2 从静态表单变成可持续更新的账户工作台。

### Phase 2：独立账户分析

目标：在已有账户快照之上新增 AI 分析与留痕。

包含：

- 独立账户分析 API
- 最新价格刷新与全账户分析
- 结果文件存储
- 最近 10 次历史查看

价值：也很高。因为它把 Stage 2 从“记账与刷新”升级为“账户经理建议层”。

### Phase 3：后续增强

可选后续方向：

- 多工作区
- 主题暴露地图
- 回撤预警
- 历史分析对比
- 半自动候选转持仓流程

## 12. 验证标准

### 12.1 持仓追踪验证

1. 手工录入 `cost_basis + shares`
2. 保存并刷新页面，字段可正确恢复
3. 读数类字段初次为空或为 0，不影响保存

### 12.2 单票写回验证

1. 持仓中已有某 ticker
2. 在 Analysis tab 查询该 ticker
3. 分析完成后自动刷新该持仓的最新价、市值、盈亏、权重
4. 账户概览同步更新
5. 页面给出明确写回反馈

### 12.3 非命中持仓验证

1. 查询一个不在持仓中的 ticker
2. 分析完成后不修改现有持仓
3. 页面明确提示未命中 Stage 2 持仓

### 12.4 账户分析验证

1. 手动触发账户分析
2. 能拿到结构化账户分析结果
3. 结果写入文件夹
4. 历史列表最多展示最近 10 次
5. 新分析写入后，旧结果按策略清理

## 13. 关键取舍

1. **继续用增强后的单工作区模型，而不是立刻做多表规范化账本**  
   因为当前目标是低摩擦、可持续使用，而不是构建完整券商级 PMS。

2. **单票写回只刷新结果字段，不碰人工真值字段**  
   因为 `cost_basis / shares / nav / cash` 才是用户维护的账户真实输入。

3. **账户分析单独触发，而不是每次单票后自动触发**  
   因为用户已经明确希望它是独立入口，避免每次查询都增加成本。

4. **分析结果文件化保留最近 10 次，而不是全部塞入数据库**  
   因为这更适合追溯和人工查看，也更符合本地工具的使用方式。

5. **在继续加功能前先做最小前端清理**  
   因为当前页面已经出现过 tab 嵌套和旧函数引用问题，不清理会继续放大维护成本。
