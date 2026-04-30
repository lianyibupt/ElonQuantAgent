# Stage2 账户分析集成 Core Skill 设计

日期：2026-04-29  
项目：ElonQuantAgent  
范围：将 `core_skill` 能力集成到 Stage2 账户分析链路，并在账户分析页面独立区块展示结果，且不破坏现有分析能力。

## 1. 目标

在用户点击“Run Account Analysis”后，系统除现有账户级分析外，自动对工作区全部持仓执行 `core_skill` 风格单票分析，并输出独立展示区块：

- 行动清单（按 `add / trim / hold` 分组）
- 评分表（每票关键分数字段）

同时满足：

- 股票行情获取统一使用 `web_interface_new` 现有数据源配置与切换能力
- 数据拉取每个标的一次，推理串行执行
- 结果写入账户分析 artifacts，支持历史回看
- 不改坏现有账户分析能力与接口兼容性

## 2. 已确认约束

1. 集成页面：`demo_new.html`（账户分析页面）
2. 触发方式：运行账户分析时自动执行
3. 覆盖范围：对所有持仓执行并汇总
4. 执行模式：行情单票单次获取；LLM 推理串行
5. 时间窗口：固定 `1d` + 最近 `90` 天
6. 展示形态：行动清单 + 评分表组合
7. 结果留存：写入 artifacts，支持历史回看
8. 失败语义：暂不使用降级策略，任一标的失败则本次账户分析整体失败

## 3. 架构设计

### 3.1 总体方案

采用“进程内桥接层”方案：新增一个独立服务模块（建议 `services/core_skill_account_bridge.py`），由 `web/web_interface_new.py` 的 `/api/account-analysis` 路由在主账户分析后调用。

桥接层职责：

1. 遍历 `workspace.positions`
2. 统一通过 `analyzer.fetch_market_data(...)` 拉取每票 `1d/90d` 数据
3. 串行执行 core-skill 风格单票分析
4. 聚合为 `core_skill_block`
5. 返回给路由并并入 API 响应与 artifacts

### 3.2 关键边界

- 不通过子进程调用 `core_skill.py`，避免多进程参数编排与双配置分叉
- 不新增独立行情抓取入口，避免与现有 `web_interface_new` 配置不一致
- 不改写原 `analysis` 结构，仅新增并行字段 `core_skill_block`

## 4. 数据流

1. `POST /api/account-analysis`
2. 刷新工作区价格并保存（现有逻辑）
3. 执行账户级分析 LLM（现有逻辑）
4. 调用桥接层生成 `core_skill_block`
5. 组合返回：`analysis + analysis_markdown + core_skill_block`
6. artifacts json 落盘包含 `core_skill_block`
7. `GET /api/account-analysis/history` 读取历史时透传 `core_skill_block`

## 5. 数据结构

新增响应字段：

```json
{
  "core_skill_block": {
    "enabled": true,
    "generated_at": "2026-04-29T12:00:00Z",
    "summary": {
      "total_positions": 0,
      "processed_positions": 0,
      "timeframe": "1d",
      "lookback_days": 90
    },
    "actions": {
      "add": [],
      "trim": [],
      "hold": []
    },
    "score_table": [
      {
        "ticker": "AAPL",
        "decision": "...",
        "recommended_action": "...",
        "trend_score": 0,
        "entry_score": 0,
        "volatility_score": 0,
        "risk_reward_ratio": "...",
        "suggested_position_range": "...",
        "justification": "..."
      }
    ]
  }
}
```

`actions` 与 `score_table` 都由单票分析结果聚合得到，便于前端分别渲染行动视图与评分视图。

## 6. 前端展示

位置：`templates/demo_new.html` Stage2 区域新增 `Core Skill 组合洞察` 面板（独立区块）。

展示规则：

1. 行动清单（add/trim/hold）
2. 评分表（ticker 与关键分数）
3. 无结果时显示占位提示
4. 历史记录切换时同步刷新该区块

兼容策略：

- 原账户分析 UI 保持不变
- 新增渲染函数 `renderCoreSkillBlock(...)`
- `runAccountAnalysis()` 成功后调用新渲染函数
- `loadAccountAnalysisHistory()` 仅维护历史列表，不改变现有交互

## 7. 失败与一致性

当前版本按“严格模式”执行：

- 任一持仓 core_skill 执行失败，则 `/api/account-analysis` 失败返回
- 不写入本次 artifacts，避免半成品历史记录
- 由前端统一展示失败信息

## 8. 测试策略

### 8.1 后端测试（`tests/test_account_analysis_api.py`）

新增/扩展用例：

1. 成功路径：
   - 多持仓输入
   - 验证 `core_skill_block.actions` 与 `score_table` 存在
   - 验证调用参数固定为 `1d/90d`
   - 验证行情通过 `analyzer.fetch_market_data` 路径
2. 失败路径：
   - 某一持仓数据为空或分析异常
   - `/api/account-analysis` 返回失败
3. 历史路径：
   - history 返回项包含 `core_skill_block`

### 8.2 前端模板测试（`tests/test_demo_new_stage2_template.py`）

新增断言：

- 存在 core skill 独立区块容器
- 存在 `renderCoreSkillBlock` 函数
- `runAccountAnalysis()` 内调用 core 区块渲染

## 9. 变更清单（预期）

1. `services/core_skill_account_bridge.py`（新）
2. `web/web_interface_new.py`（改）
3. `templates/demo_new.html`（改）
4. `tests/test_account_analysis_api.py`（改）
5. `tests/test_demo_new_stage2_template.py`（改）

## 10. 非目标

- 不调整原单票分析主链路
- 不引入并发执行
- 不新增后台任务系统
- 不接入多工作区批处理
- 不在本轮引入降级策略
