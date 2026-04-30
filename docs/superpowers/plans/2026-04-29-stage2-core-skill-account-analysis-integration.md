# Stage2 Core Skill Account Analysis Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate core-skill-style per-position analysis into Stage2 account-analysis API and render it as an independent block on the Stage2 page, while preserving existing account-analysis behavior.

**Architecture:** Add a dedicated bridge service that orchestrates per-position analysis for all workspace positions using the existing `web_interface_new` market-data configuration and fetch path. Extend `/api/account-analysis` to call this bridge, attach `core_skill_block` to response/artifacts/history, and add an isolated frontend rendering panel for action groups and score table. Keep old response fields unchanged to minimize regression risk.

**Tech Stack:** Flask, Python unittest, pandas DataFrame, existing TradingGraph pipeline, Bootstrap/vanilla JS template rendering.

---

## File Structure & Responsibilities

- Create: `services/core_skill_account_bridge.py`
  - Responsibility: Orchestrate all-position core-skill block generation, enforce `1d/90d`, serial execution, action grouping, and deterministic output shape.
- Modify: `web/web_interface_new.py`
  - Responsibility: Wire bridge call into `/api/account-analysis`, adapt existing analyzer call as injectable callback, and include `core_skill_block` in API response and artifact payload.
- Modify: `services/account_analysis.py`
  - Responsibility: Include `core_skill_block` in history list entries so Stage2 history can render past core-skill panels.
- Modify: `templates/demo_new.html`
  - Responsibility: Add isolated Stage2 “Core Skill 组合洞察” panel and JS renderer without changing existing account-analysis panel semantics.
- Create: `tests/test_core_skill_account_bridge.py`
  - Responsibility: Unit-test bridge orchestration (all positions processed, action grouping, strict failure behavior, fixed lookback/timeframe).
- Modify: `tests/test_account_analysis_api.py`
  - Responsibility: TDD coverage for `/api/account-analysis` returning `core_skill_block` and history including it.
- Modify: `tests/test_demo_new_stage2_template.py`
  - Responsibility: Ensure template contains dedicated core-skill surface and renderer hook.

---

### Task 1: Add failing bridge and API tests first (RED)

**Files:**
- Create: `tests/test_core_skill_account_bridge.py`
- Modify: `tests/test_account_analysis_api.py`

- [ ] **Step 1: Write failing bridge unit tests**

```python
import unittest
from datetime import datetime, timezone
import pandas as pd

from services.core_skill_account_bridge import build_core_skill_block


class CoreSkillAccountBridgeTests(unittest.TestCase):
    def test_build_core_skill_block_groups_actions_and_scores_for_all_positions(self):
        workspace = {
            "account_state": {"nav": 100000, "cash": 20000},
            "positions": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
            "candidates": []
        }

        market_df = pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2026-04-28", "2026-04-29"]),
                "Open": [100, 101],
                "High": [101, 102],
                "Low": [99, 100],
                "Close": [100.5, 101.5],
            }
        )

        fetch_calls = []
        def fetch_market_data(symbol, interval, start_dt, end_dt, market_data_source=None):
            fetch_calls.append((symbol, interval, market_data_source))
            return market_df

        def analyze_position(ticker, df, workspace_payload):
            return {
                "single_name_score": {
                    "decision": "持有",
                    "recommended_action": "观察",
                    "trend_score": 70,
                    "entry_score": 55,
                    "volatility_score": 45,
                    "risk_reward_ratio": "1.5:1",
                    "suggested_position_range": "3%-5%",
                    "justification": f"{ticker} 结构正常",
                }
            }

        block = build_core_skill_block(
            workspace_name="default",
            workspace=workspace,
            fetch_market_data=fetch_market_data,
            analyze_position=analyze_position,
            market_data_source="yfinance",
            now=datetime(2026, 4, 29, tzinfo=timezone.utc),
        )

        self.assertEqual(block["summary"]["total_positions"], 2)
        self.assertEqual(block["summary"]["processed_positions"], 2)
        self.assertEqual(block["summary"]["timeframe"], "1d")
        self.assertEqual(block["summary"]["lookback_days"], 90)
        self.assertEqual(len(block["score_table"]), 2)
        self.assertEqual(fetch_calls[0][1], "1d")

    def test_build_core_skill_block_raises_when_any_position_fails(self):
        workspace = {"account_state": {}, "positions": [{"ticker": "AAPL"}], "candidates": []}

        def fetch_market_data(symbol, interval, start_dt, end_dt, market_data_source=None):
            return pd.DataFrame()

        def analyze_position(ticker, df, workspace_payload):
            return {}

        with self.assertRaises(ValueError):
            build_core_skill_block(
                workspace_name="default",
                workspace=workspace,
                fetch_market_data=fetch_market_data,
                analyze_position=analyze_position,
                market_data_source="yfinance",
            )
```

- [ ] **Step 2: Run bridge tests to verify fail**

Run: `python -m unittest tests.test_core_skill_account_bridge -v`  
Expected: FAIL with import error / missing function because bridge module is not implemented yet.

- [ ] **Step 3: Add failing API assertions for core_skill_block**

```python
# in tests/test_account_analysis_api.py, success-case test
self.assertIn("core_skill_block", payload)
self.assertEqual(payload["core_skill_block"]["summary"]["timeframe"], "1d")
self.assertEqual(payload["core_skill_block"]["summary"]["lookback_days"], 90)
self.assertGreaterEqual(payload["core_skill_block"]["summary"]["processed_positions"], 1)
self.assertIn("score_table", payload["core_skill_block"])
```

- [ ] **Step 4: Run focused API test to verify fail**

Run: `python -m unittest tests.test_account_analysis_api.AccountAnalysisApiTests.test_post_account_analysis_refreshes_prices_runs_llm_and_saves_artifacts -v`  
Expected: FAIL because `core_skill_block` does not exist yet.

- [ ] **Step 5: Commit RED tests**

```bash
git add tests/test_core_skill_account_bridge.py tests/test_account_analysis_api.py
git commit -m "test: add failing coverage for stage2 core skill block"
```

---

### Task 2: Implement bridge service minimally (GREEN)

**Files:**
- Create: `services/core_skill_account_bridge.py`
- Test: `tests/test_core_skill_account_bridge.py`

- [ ] **Step 1: Implement bridge module with strict semantics**

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


def _normalize_action(action: str) -> str:
    text = str(action or "").strip().lower()
    if any(token in text for token in ["buy", "add", "加仓", "买入"]):
        return "add"
    if any(token in text for token in ["trim", "sell", "reduce", "减仓", "卖出"]):
        return "trim"
    return "hold"


def _score_row_from_result(ticker: str, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    score = (analysis_result or {}).get("single_name_score", {}) or {}
    return {
        "ticker": ticker,
        "decision": str(score.get("decision", "N/A")),
        "recommended_action": str(score.get("recommended_action", "观察")),
        "trend_score": score.get("trend_score", 0),
        "entry_score": score.get("entry_score", 0),
        "volatility_score": score.get("volatility_score", 0),
        "risk_reward_ratio": str(score.get("risk_reward_ratio", "N/A")),
        "suggested_position_range": str(score.get("suggested_position_range", "N/A")),
        "justification": str(score.get("justification", "")),
    }


def build_core_skill_block(
    workspace_name: str,
    workspace: Dict[str, Any],
    fetch_market_data: Callable[..., pd.DataFrame],
    analyze_position: Callable[[str, pd.DataFrame, Dict[str, Any]], Dict[str, Any]],
    market_data_source: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    start_dt = current - timedelta(days=90)

    positions = (workspace or {}).get("positions", []) or []
    actions = {"add": [], "trim": [], "hold": []}
    score_table: List[Dict[str, Any]] = []

    for position in positions:
        ticker = str((position or {}).get("ticker", "")).strip().upper()
        if not ticker:
            raise ValueError("Core skill analysis failed: empty ticker in workspace positions")

        df = fetch_market_data(ticker, "1d", start_dt, current, market_data_source=market_data_source)
        if df is None or getattr(df, "empty", True):
            raise ValueError(f"Core skill analysis failed for {ticker}: no market data")

        result = analyze_position(ticker, df, workspace)
        row = _score_row_from_result(ticker, result)
        score_table.append(row)

        bucket = _normalize_action(row.get("recommended_action", ""))
        actions[bucket].append({
            "ticker": ticker,
            "action": row.get("recommended_action", "观察"),
            "reason": row.get("justification", ""),
        })

    return {
        "enabled": True,
        "generated_at": current.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "workspace_name": str(workspace_name or "default"),
        "summary": {
            "total_positions": len(positions),
            "processed_positions": len(score_table),
            "timeframe": "1d",
            "lookback_days": 90,
        },
        "actions": actions,
        "score_table": score_table,
    }
```

- [ ] **Step 2: Run bridge tests to verify pass**

Run: `python -m unittest tests.test_core_skill_account_bridge -v`  
Expected: PASS for both new unit tests.

- [ ] **Step 3: Commit bridge implementation**

```bash
git add services/core_skill_account_bridge.py tests/test_core_skill_account_bridge.py
git commit -m "feat: add core skill account bridge service"
```

---

### Task 3: Wire bridge into account-analysis API and history

**Files:**
- Modify: `web/web_interface_new.py`
- Modify: `services/account_analysis.py`
- Test: `tests/test_account_analysis_api.py`

- [ ] **Step 1: Add backend helper in web interface**

```python
# web/web_interface_new.py imports
from services.core_skill_account_bridge import build_core_skill_block


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
```

- [ ] **Step 2: Attach core_skill_block into `/api/account-analysis` response and artifacts payload**

```python
core_skill_block = _run_core_skill_account_block(
    workspace_name=workspace_name,
    workspace=saved_workspace,
    market_data_source=market_data_source,
)

analysis_payload["core_skill_block"] = core_skill_block

# existing return payload
"core_skill_block": core_skill_block,
```

- [ ] **Step 3: Extend account-analysis history entries with core_skill_block**

```python
# services/account_analysis.py in _collect_account_analysis_history entries.append(...)
"core_skill_block": payload.get("core_skill_block", {}),
```

- [ ] **Step 4: Run API tests to verify green**

Run: `python -m unittest tests.test_account_analysis_api -v`  
Expected: PASS including new `core_skill_block` assertions.

- [ ] **Step 5: Commit backend integration**

```bash
git add web/web_interface_new.py services/account_analysis.py tests/test_account_analysis_api.py
git commit -m "feat: include core skill block in stage2 account analysis api"
```

---

### Task 4: Add isolated Stage2 core-skill UI block

**Files:**
- Modify: `templates/demo_new.html`
- Modify: `tests/test_demo_new_stage2_template.py`

- [ ] **Step 1: Add Stage2 panel markup for core-skill block**

```html
<div class="panel" id="workspaceCoreSkillPanel">
  <h4 class="panel-title"><i class="fas fa-layer-group"></i> Core Skill 组合洞察</h4>
  <div id="workspaceCoreSkillSummary" class="text-muted mb-2">暂无 Core Skill 分析结果。</div>

  <div class="row g-3 mb-3">
    <div class="col-md-4"><div class="card card-body"><h6>加仓</h6><ul id="workspaceCoreSkillActionsAdd" class="mb-0"></ul></div></div>
    <div class="col-md-4"><div class="card card-body"><h6>减仓</h6><ul id="workspaceCoreSkillActionsTrim" class="mb-0"></ul></div></div>
    <div class="col-md-4"><div class="card card-body"><h6>观察</h6><ul id="workspaceCoreSkillActionsHold" class="mb-0"></ul></div></div>
  </div>

  <div class="table-responsive">
    <table class="table table-sm align-middle" id="workspaceCoreSkillScoreTable">
      <thead>
        <tr>
          <th>Ticker</th><th>Decision</th><th>Action</th><th>Trend</th><th>Entry</th><th>Volatility</th><th>R/R</th><th>Position</th>
        </tr>
      </thead>
      <tbody id="workspaceCoreSkillScoreTableBody"></tbody>
    </table>
  </div>
</div>
```

- [ ] **Step 2: Add JS renderer and wire to run/history flow**

```javascript
function renderCoreSkillBlock(coreSkillBlock) {
  const summaryEl = document.getElementById('workspaceCoreSkillSummary');
  const addEl = document.getElementById('workspaceCoreSkillActionsAdd');
  const trimEl = document.getElementById('workspaceCoreSkillActionsTrim');
  const holdEl = document.getElementById('workspaceCoreSkillActionsHold');
  const tableBody = document.getElementById('workspaceCoreSkillScoreTableBody');
  if (!summaryEl || !addEl || !trimEl || !holdEl || !tableBody) return;

  const block = coreSkillBlock || {};
  const summary = block.summary || {};
  summaryEl.textContent = block.enabled
    ? `已处理 ${summary.processed_positions || 0}/${summary.total_positions || 0} 持仓（${summary.timeframe || '1d'}，${summary.lookback_days || 90}天）`
    : '暂无 Core Skill 分析结果。';

  const renderActionList = (target, items) => {
    target.innerHTML = '';
    const list = Array.isArray(items) ? items : [];
    if (!list.length) {
      const li = document.createElement('li');
      li.className = 'text-muted';
      li.textContent = '无';
      target.appendChild(li);
      return;
    }
    list.forEach(item => {
      const li = document.createElement('li');
      li.textContent = `${item.ticker || 'N/A'}：${item.reason || item.action || '—'}`;
      target.appendChild(li);
    });
  };

  const actions = block.actions || {};
  renderActionList(addEl, actions.add);
  renderActionList(trimEl, actions.trim);
  renderActionList(holdEl, actions.hold);

  tableBody.innerHTML = '';
  const rows = Array.isArray(block.score_table) ? block.score_table : [];
  rows.forEach(row => {
    const tr = document.createElement('tr');
    [
      row.ticker, row.decision, row.recommended_action,
      row.trend_score, row.entry_score, row.volatility_score,
      row.risk_reward_ratio, row.suggested_position_range
    ].forEach(value => {
      const td = document.createElement('td');
      td.textContent = value ?? '—';
      tr.appendChild(td);
    });
    tableBody.appendChild(tr);
  });
}

// runAccountAnalysis success branch
renderCoreSkillBlock(data.core_skill_block || (data.artifacts && data.artifacts.core_skill_block));

// loadAccountAnalysisHistory success branch
const latest = (data.history || [])[0] || {};
renderCoreSkillBlock(latest.core_skill_block || null);
```

- [ ] **Step 3: Add template-level tests for new markers**

```python
# tests/test_demo_new_stage2_template.py
for marker in [
    'workspaceCoreSkillPanel',
    'workspaceCoreSkillSummary',
    'workspaceCoreSkillActionsAdd',
    'workspaceCoreSkillActionsTrim',
    'workspaceCoreSkillActionsHold',
    'workspaceCoreSkillScoreTableBody',
    'function renderCoreSkillBlock(coreSkillBlock)',
]:
    with self.subTest(marker=marker):
        self.assertIn(marker, self.template_text)
```

- [ ] **Step 4: Run template tests**

Run: `python -m unittest tests.test_demo_new_stage2_template -v`  
Expected: PASS with all marker checks.

- [ ] **Step 5: Commit frontend core-skill block**

```bash
git add templates/demo_new.html tests/test_demo_new_stage2_template.py
git commit -m "feat: add stage2 core skill insights panel"
```

---

### Task 5: End-to-end verification and cleanup

**Files:**
- Modify (if needed): `web/web_interface_new.py`, `tests/*`

- [ ] **Step 1: Run focused backend test suite**

Run: `python -m unittest tests.test_core_skill_account_bridge tests.test_account_analysis_api -v`  
Expected: PASS.

- [ ] **Step 2: Run template test suite**

Run: `python -m unittest tests.test_demo_new_stage2_template -v`  
Expected: PASS.

- [ ] **Step 3: Run manual smoke for Stage2 account analysis path**

Run server: `python run.py --port 5001`  
Manual checks:
- Open Stage2 tab in `demo_new.html`
- Click `Run Account Analysis`
- Confirm core-skill panel renders actions + score table
- Confirm existing account analysis status/history still updates

Expected: main account-analysis behavior unchanged, with additional core-skill panel.

- [ ] **Step 4: Commit verification fixes (if any)**

```bash
git add web/web_interface_new.py services/account_analysis.py templates/demo_new.html tests/test_core_skill_account_bridge.py tests/test_account_analysis_api.py tests/test_demo_new_stage2_template.py
git commit -m "test: verify stage2 core skill integration end-to-end"
```

---

## Self-Review

### Spec coverage

- Core-skill integrated into Stage2 account-analysis flow: covered in Task 3.
- Uses web interface market-data config/fetch path only: covered in Task 3 (`analyzer.fetch_market_data`).
- All positions processed, serial LLM run, 1d/90d: covered in Task 2 + Task 3 + bridge tests.
- Independent frontend block showing action list + score table: covered in Task 4.
- Persisted and surfaced in history: covered in Task 3 (`services/account_analysis.py` + API tests).
- Strict failure semantics (no degradation): covered in Task 2 (`ValueError`) and Task 3 API path.

### Placeholder scan

- No TODO/TBD placeholders.
- Every code step includes concrete snippets.
- Every test step includes command and expected result.

### Type/signature consistency

- `build_core_skill_block(...)` signature is consistent between Task 2 implementation and Task 3 call site.
- `core_skill_block.summary` field names (`timeframe`, `lookback_days`, `processed_positions`) used consistently in API tests and frontend rendering.
