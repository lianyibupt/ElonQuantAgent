# Stage 2 Account Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Stage 2 from a saved context form into a persistent account workspace with derived holding metrics, post-analysis write-back, and a separate account-analysis workflow with retained history.

**Architecture:** Keep the current Flask + SQLite + single-page template architecture, but move Stage 2 business logic out of `web/web_interface_new.py` into focused service modules. Phase 1 delivers a reliable account-tracking loop and cleans the duplicated frontend analysis code. Phase 2 adds a manual account-analysis route that refreshes prices, asks the LLM for a portfolio-level assessment, writes JSON/Markdown artifacts, and exposes a small recent-history API.

**Tech Stack:** Flask, SQLite, pandas, yfinance, existing LLM provider wrapper, vanilla JavaScript, Python `unittest`

---

## File map

### Existing files to modify
- `templates/demo_new.html:1492-1594` — current Stage 2 workspace markup; extend position columns, add account summary cards, account-analysis controls, and history list.
- `templates/demo_new.html:1683-1915` — current Stage 2 state/render/save JS; replace `market_value`-only inputs with manual truth fields plus derived read-only fields.
- `templates/demo_new.html:2463-2552` — stale duplicate `runAnalysis()` block to delete.
- `templates/demo_new.html:3090-3204` — active `runAnalysis()` block; keep this as the single analysis entry and add Stage 2 write-back/account-analysis UI hooks.
- `web/web_interface_new.py:70-168` — current default workspace schema and payload normalization; replace with imports from a focused Stage 2 service.
- `web/web_interface_new.py:794-892` — analysis response formatting; add `latest_price` and write-back metadata fields that the frontend and API tests can assert.
- `web/web_interface_new.py:1023-1046` — Stage 2 load/save routes; keep endpoints stable but return the richer workspace shape.
- `web/web_interface_new.py:1198-1495` — `/api/analyze`; add post-analysis write-back and return a small `workspace_writeback` payload.
- `services/database.py:162-190` — current `stage2_workspaces` table; keep the table, optionally add a small index table for account-analysis history only if the history API needs metadata beyond file scanning.
- `services/database.py:855-920` — Stage 2 CRUD helpers; continue storing JSON blobs, but ensure richer positions/account fields round-trip unchanged.
- `core/trading_graph.py:399-474` — current dashboard aggregation uses `market_value`; leave Phase 1 compatible by sending derived `market_value` in positions.

### New files to create
- `services/stage2_workspace.py` — payload normalization, per-position derived metrics, account aggregation, and post-analysis write-back helpers.
- `services/account_analysis.py` — account-analysis prompt builder, latest-price refresh helper, artifact writer, retention, and history listing.
- `tests/__init__.py` — test package marker.
- `tests/test_stage2_workspace_service.py` — unit tests for normalization, derived fields, and aggregate metrics.
- `tests/test_stage2_workspace_api.py` — Flask API tests for Stage 2 GET/POST routes.
- `tests/test_stage2_writeback.py` — Flask/route tests for `/api/analyze` write-back behavior.
- `tests/test_account_analysis_service.py` — unit tests for account-analysis artifact retention and summary rendering.
- `tests/test_demo_new_stage2_template.py` — template-level regression tests for one `runAnalysis()` definition and the new Stage 2 / Account Analysis anchors.

### Execution order
1. Phase 1 foundation: service module + backend schema normalization.
2. Phase 1 UI cleanup + richer Stage 2 editor.
3. Phase 1 analysis write-back.
4. Phase 2 account-analysis service + APIs.
5. Phase 2 frontend controls/history.
6. Final verification.

---

## Phase 1 — Account tracking loop

### Task 1: Create Stage 2 workspace service and test harness

**Files:**
- Create: `services/stage2_workspace.py`
- Create: `tests/__init__.py`
- Create: `tests/test_stage2_workspace_service.py`

- [ ] **Step 1: Write the failing service tests**

```python
import unittest

from services.stage2_workspace import (
    normalize_workspace_payload,
    recalculate_account_state,
    refresh_position_after_analysis,
)


class Stage2WorkspaceServiceTests(unittest.TestCase):
    def test_normalize_workspace_payload_keeps_manual_truth_and_zeroes_missing_derived_fields(self):
        workspace = normalize_workspace_payload({
            "account_state": {"nav": "100000", "cash": "25000", "current_drawdown": "8"},
            "positions": [{
                "ticker": "mstr",
                "book_type": "核心仓",
                "cost_basis": "152.4",
                "shares": "120",
                "tracking_status": "持续跟踪",
                "factor_tags": "btc-beta, proxy",
            }],
            "candidates": [{"ticker": "crcl", "factor_tags": "fintech, growth"}],
        })
        self.assertEqual(workspace["positions"][0]["ticker"], "MSTR")
        self.assertEqual(workspace["positions"][0]["cost_basis"], 152.4)
        self.assertEqual(workspace["positions"][0]["shares"], 120.0)
        self.assertEqual(workspace["positions"][0]["market_value"], 0.0)
        self.assertEqual(workspace["positions"][0]["factor_tags"], ["btc-beta", "proxy"])

    def test_recalculate_account_state_derives_market_value_weight_and_exposure(self):
        workspace = normalize_workspace_payload({
            "account_state": {"nav": 100000, "cash": 25000, "current_drawdown": 8},
            "positions": [
                {"ticker": "MSTR", "book_type": "核心仓", "cost_basis": 100, "shares": 100, "latest_price": 150},
                {"ticker": "CRCL", "book_type": "战术仓", "cost_basis": 50, "shares": 200, "latest_price": 40},
            ],
        })
        recalculated = recalculate_account_state(workspace)
        self.assertEqual(recalculated["positions"][0]["market_value"], 15000.0)
        self.assertEqual(recalculated["positions"][0]["unrealized_pnl"], 5000.0)
        self.assertEqual(recalculated["positions"][0]["position_weight"], 15.0)
        self.assertEqual(recalculated["account_state"]["total_market_value"], 23000.0)
        self.assertEqual(recalculated["account_state"]["gross_exposure"], 23.0)
        self.assertEqual(recalculated["account_state"]["largest_position"], "MSTR")

    def test_refresh_position_after_analysis_updates_only_matching_holding(self):
        workspace = normalize_workspace_payload({
            "account_state": {"nav": 100000, "cash": 20000},
            "positions": [{"ticker": "MSTR", "book_type": "核心仓", "cost_basis": 100, "shares": 50}],
        })
        updated, writeback = refresh_position_after_analysis(
            workspace,
            asset="MSTR",
            latest_price=125.0,
            analysis_summary="趋势仍强",
            updated_at="2026-04-28T18:00:00",
        )
        self.assertTrue(writeback["updated"])
        self.assertEqual(updated["positions"][0]["latest_price"], 125.0)
        self.assertEqual(updated["positions"][0]["last_analysis_summary"], "趋势仍强")
        self.assertEqual(updated["positions"][0]["unrealized_pnl_pct"], 25.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the service tests and verify they fail**

Run: `python -m unittest tests.test_stage2_workspace_service -v`
Expected: `ImportError: No module named 'services.stage2_workspace'`

- [ ] **Step 3: Write the minimal Stage 2 workspace service**

```python
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Tuple


def _coerce_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _normalize_tags(value: Any) -> List[str]:
    if isinstance(value, str):
        return [tag.strip() for tag in value.split(",") if tag.strip()]
    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]
    return []


def build_default_workspace() -> Dict[str, Any]:
    return {
        "workspace_name": "default",
        "account_state": {
            "nav": 0.0,
            "cash": 0.0,
            "total_market_value": 0.0,
            "cash_pct": 0.0,
            "gross_exposure": 0.0,
            "core_exposure": 0.0,
            "tactical_exposure": 0.0,
            "largest_position": "",
            "largest_position_weight": 0.0,
            "position_count": 0,
            "current_drawdown": 0.0,
            "last_aggregated_at": None,
        },
        "positions": [],
        "candidates": [],
        "notes": "",
        "created_at": None,
        "updated_at": None,
    }


def normalize_workspace_payload(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    base = build_default_workspace()
    payload = payload or {}
    account = payload.get("account_state") or {}
    positions = payload.get("positions") or []
    candidates = payload.get("candidates") or []
    base["notes"] = str(payload.get("notes", "")).strip()
    base["account_state"].update({
        "nav": _coerce_float(account.get("nav")),
        "cash": _coerce_float(account.get("cash")),
        "current_drawdown": _coerce_float(account.get("current_drawdown")),
    })
    normalized_positions = []
    for position in positions:
        ticker = str(position.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        normalized_positions.append({
            "ticker": ticker,
            "book_type": str(position.get("book_type", "观察")).strip() or "观察",
            "cost_basis": _coerce_float(position.get("cost_basis")),
            "shares": _coerce_float(position.get("shares")),
            "tracking_status": str(position.get("tracking_status", "持续跟踪")).strip() or "持续跟踪",
            "factor_tags": _normalize_tags(position.get("factor_tags")),
            "notes": str(position.get("notes", "")).strip(),
            "latest_price": _coerce_float(position.get("latest_price")),
            "market_value": _coerce_float(position.get("market_value")),
            "position_weight": _coerce_float(position.get("position_weight")),
            "unrealized_pnl": _coerce_float(position.get("unrealized_pnl")),
            "unrealized_pnl_pct": _coerce_float(position.get("unrealized_pnl_pct")),
            "price_source": str(position.get("price_source", "")),
            "last_price_update_at": position.get("last_price_update_at"),
            "last_analyzed_at": position.get("last_analyzed_at"),
            "last_analysis_summary": str(position.get("last_analysis_summary", "")).strip(),
        })
    base["positions"] = normalized_positions
    base["candidates"] = [
        {
            "ticker": str(candidate.get("ticker", "")).strip().upper(),
            "factor_tags": _normalize_tags(candidate.get("factor_tags")),
            "notes": str(candidate.get("notes", "")).strip(),
            "last_screened_at": candidate.get("last_screened_at"),
        }
        for candidate in candidates
        if str(candidate.get("ticker", "")).strip()
    ]
    return recalculate_account_state(base)


def recalculate_account_state(workspace: Dict[str, Any]) -> Dict[str, Any]:
    workspace = deepcopy(workspace)
    nav = _coerce_float(workspace["account_state"].get("nav"))
    cash = _coerce_float(workspace["account_state"].get("cash"))
    total_market_value = 0.0
    core_market_value = 0.0
    tactical_market_value = 0.0
    largest_ticker = ""
    largest_value = 0.0
    for position in workspace["positions"]:
        latest_price = _coerce_float(position.get("latest_price"))
        shares = _coerce_float(position.get("shares"))
        cost_basis = _coerce_float(position.get("cost_basis"))
        market_value = round(latest_price * shares, 2) if latest_price > 0 and shares > 0 else 0.0
        unrealized_pnl = round((latest_price - cost_basis) * shares, 2) if latest_price > 0 and shares > 0 else 0.0
        unrealized_pnl_pct = round(((latest_price - cost_basis) / cost_basis) * 100, 2) if latest_price > 0 and shares > 0 and cost_basis > 0 else 0.0
        position_weight = round((market_value / nav) * 100, 2) if nav > 0 and market_value > 0 else 0.0
        position.update({
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "position_weight": position_weight,
        })
        total_market_value += market_value
        if position.get("book_type") == "核心仓":
            core_market_value += market_value
        if position.get("book_type") == "战术仓":
            tactical_market_value += market_value
        if market_value > largest_value:
            largest_ticker = position["ticker"]
            largest_value = market_value
    workspace["account_state"].update({
        "total_market_value": round(total_market_value, 2),
        "cash_pct": round((cash / nav) * 100, 2) if nav > 0 else 0.0,
        "gross_exposure": round((total_market_value / nav) * 100, 2) if nav > 0 else 0.0,
        "core_exposure": round((core_market_value / nav) * 100, 2) if nav > 0 else 0.0,
        "tactical_exposure": round((tactical_market_value / nav) * 100, 2) if nav > 0 else 0.0,
        "largest_position": largest_ticker,
        "largest_position_weight": round((largest_value / nav) * 100, 2) if nav > 0 else 0.0,
        "position_count": sum(1 for position in workspace["positions"] if _coerce_float(position.get("shares")) > 0),
        "last_aggregated_at": datetime.utcnow().isoformat(timespec="seconds"),
    })
    return workspace


def refresh_position_after_analysis(workspace: Dict[str, Any], asset: str, latest_price: float, analysis_summary: str, updated_at: str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    workspace = deepcopy(workspace)
    updated_at = updated_at or datetime.utcnow().isoformat(timespec="seconds")
    for position in workspace["positions"]:
        if position.get("ticker") != str(asset).strip().upper():
            continue
        position["latest_price"] = round(float(latest_price), 4)
        position["price_source"] = "analysis"
        position["last_price_update_at"] = updated_at
        position["last_analyzed_at"] = updated_at
        position["last_analysis_summary"] = analysis_summary.strip()
        updated_workspace = recalculate_account_state(workspace)
        return updated_workspace, {"updated": True, "ticker": position["ticker"], "message": f"已更新 {position['ticker']} 的最新价格与浮盈亏"}
    return workspace, {"updated": False, "ticker": str(asset).strip().upper(), "message": "当前标的不在 Stage 2 持仓中，未写回账户"}
```

- [ ] **Step 4: Run the service tests and verify they pass**

Run: `python -m unittest tests.test_stage2_workspace_service -v`
Expected: `Ran 3 tests` and `OK`

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/test_stage2_workspace_service.py services/stage2_workspace.py
git commit -m "feat: add stage2 workspace metrics service"
```

### Task 2: Wire the richer Stage 2 schema into load/save APIs

**Files:**
- Modify: `web/web_interface_new.py:70-168`
- Modify: `web/web_interface_new.py:1023-1046`
- Modify: `services/database.py:855-920`
- Create: `tests/test_stage2_workspace_api.py`

- [ ] **Step 1: Write the failing API tests**

```python
import unittest

from web.web_interface_new import app, db_manager


class Stage2WorkspaceApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        db_manager.save_stage2_workspace(workspace_name="default", account_state={}, positions=[], candidates=[], notes="")

    def test_get_stage2_workspace_returns_rich_default_shape(self):
        response = self.client.get("/api/stage2-workspace")
        payload = response.get_json()
        self.assertTrue(payload["success"])
        position = payload["workspace"]["positions"]
        self.assertEqual(position, [])
        self.assertIn("total_market_value", payload["workspace"]["account_state"])

    def test_post_stage2_workspace_normalizes_cost_basis_and_shares(self):
        response = self.client.post("/api/stage2-workspace", json={
            "account_state": {"nav": "100000", "cash": "30000", "current_drawdown": "4"},
            "positions": [{
                "ticker": "mstr",
                "book_type": "核心仓",
                "cost_basis": "120",
                "shares": "10",
                "tracking_status": "持续跟踪",
                "factor_tags": "btc-beta, proxy",
            }],
            "candidates": [],
            "notes": "live book",
        })
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["workspace"]["positions"][0]["ticker"], "MSTR")
        self.assertEqual(payload["workspace"]["positions"][0]["shares"], 10.0)
        self.assertEqual(payload["workspace"]["positions"][0]["market_value"], 0.0)
```

- [ ] **Step 2: Run the API tests and verify they fail**

Run: `python -m unittest tests.test_stage2_workspace_api -v`
Expected: FAIL because the current payload still expects `market_value` input and the default account shape lacks derived fields.

- [ ] **Step 3: Replace inline normalization with the shared Stage 2 service**

```python
from services.stage2_workspace import build_default_workspace, normalize_workspace_payload

STAGE2_DEFAULT_WORKSPACE = build_default_workspace()


def normalize_stage2_workspace_payload(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        return normalize_workspace_payload(payload)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


@app.route('/api/stage2-workspace', methods=['GET'])
def get_stage2_workspace():
    workspace_name = request.args.get('workspace_name', 'default')
    workspace = db_manager.get_stage2_workspace(workspace_name) or build_default_workspace()
    return jsonify({"success": True, "workspace": workspace})
```

```python
def get_stage2_workspace(self, workspace_name: str = 'default') -> Optional[Dict[str, Any]]:
    with self.get_connection() as conn:
        cursor = conn.execute("SELECT * FROM stage2_workspaces WHERE workspace_name = ?", (workspace_name,))
        row = cursor.fetchone()
        if not row:
            return None
        record = dict(row)
        record['account_state'] = json.loads(record['account_state']) if record.get('account_state') else {}
        record['positions'] = json.loads(record['positions']) if record.get('positions') else []
        record['candidates'] = json.loads(record['candidates']) if record.get('candidates') else []
        return record
```

- [ ] **Step 4: Run the API tests and verify they pass**

Run: `python -m unittest tests.test_stage2_workspace_api -v`
Expected: `Ran 2 tests` and `OK`

- [ ] **Step 5: Commit**

```bash
git add web/web_interface_new.py services/database.py tests/test_stage2_workspace_api.py
git commit -m "feat: wire richer stage2 workspace schema into api"
```

### Task 3: Clean duplicate frontend JS and extend the Stage 2 position editor

**Files:**
- Modify: `templates/demo_new.html:1492-1594`
- Modify: `templates/demo_new.html:1683-1915`
- Modify: `templates/demo_new.html:2463-2552`
- Modify: `templates/demo_new.html:3090-3204`
- Create: `tests/test_demo_new_stage2_template.py`

- [ ] **Step 1: Write the failing template regression test**

```python
import unittest
from pathlib import Path


class DemoNewTemplateTests(unittest.TestCase):
    def test_stage2_template_has_single_run_analysis_and_rich_position_columns(self):
        html = Path("templates/demo_new.html").read_text(encoding="utf-8")
        self.assertEqual(html.count("function runAnalysis()"), 1)
        self.assertIn("cost_basis", html)
        self.assertIn("tracking_status", html)
        self.assertIn("workspaceAccountAnalysisBtn", html)
        self.assertIn("workspaceAnalysisHistoryList", html)
```

- [ ] **Step 2: Run the template regression test and verify it fails**

Run: `python -m unittest tests.test_demo_new_stage2_template -v`
Expected: FAIL because the file currently contains two `runAnalysis()` definitions and the new account-analysis anchors do not exist.

- [ ] **Step 3: Update the Stage 2 markup and remove the stale `runAnalysis()` block**

```html
<thead>
  <tr>
    <th>Ticker</th>
    <th>Book</th>
    <th>Cost</th>
    <th>Shares</th>
    <th>Tracking</th>
    <th>Tags</th>
    <th>Latest</th>
    <th>Value</th>
    <th>Weight</th>
    <th>PnL</th>
    <th>PnL %</th>
    <th>Updated</th>
    <th></th>
  </tr>
</thead>
<tbody id="workspacePositionsBody"></tbody>
```

```javascript
function buildEmptyPosition() {
    return {
        ticker: '',
        book_type: '核心仓',
        cost_basis: 0,
        shares: 0,
        tracking_status: '持续跟踪',
        factor_tags: [],
        notes: '',
        latest_price: 0,
        market_value: 0,
        position_weight: 0,
        unrealized_pnl: 0,
        unrealized_pnl_pct: 0,
        last_price_update_at: ''
    };
}

function updatePositionField(index, field, value) {
    if (!stage2WorkspaceState.positions[index]) return;
    if (['cost_basis', 'shares'].includes(field)) {
        const parsed = Number(value || 0);
        stage2WorkspaceState.positions[index][field] = Number.isNaN(parsed) ? 0 : parsed;
        return;
    }
    if (field === 'factor_tags') {
        stage2WorkspaceState.positions[index][field] = parseTagInput(value);
        return;
    }
    stage2WorkspaceState.positions[index][field] = field === 'ticker'
        ? String(value || '').trim().toUpperCase()
        : value;
}
```

```html
<div class="workspace-inline-actions">
  <button type="button" class="btn btn-primary" id="workspaceAccountAnalysisBtn" onclick="runAccountAnalysis()">
    <i class="fas fa-brain"></i> Run Account Analysis
  </button>
</div>
<div id="workspaceAnalysisHistoryList" class="workspace-history-list"></div>
```

- [ ] **Step 4: Run the template regression test and verify it passes**

Run: `python -m unittest tests.test_demo_new_stage2_template -v`
Expected: `Ran 1 test` and `OK`

- [ ] **Step 5: Commit**

```bash
git add templates/demo_new.html tests/test_demo_new_stage2_template.py
git commit -m "refactor: clean stage2 frontend and expand holdings editor"
```

### Task 4: Add post-analysis write-back for holdings and account aggregates

**Files:**
- Modify: `web/web_interface_new.py:794-892`
- Modify: `web/web_interface_new.py:1198-1495`
- Modify: `services/database.py:855-920`
- Modify: `services/stage2_workspace.py`
- Create: `tests/test_stage2_writeback.py`

- [ ] **Step 1: Write the failing write-back tests**

```python
import unittest
from unittest.mock import patch

from web.web_interface_new import app, db_manager


class Stage2WritebackTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        db_manager.save_stage2_workspace(
            workspace_name="default",
            account_state={"nav": 100000, "cash": 20000, "current_drawdown": 5},
            positions=[{"ticker": "MSTR", "book_type": "核心仓", "cost_basis": 100, "shares": 10, "tracking_status": "持续跟踪", "factor_tags": []}],
            candidates=[],
            notes="",
        )

    @patch("web.web_interface_new.analyzer.fetch_market_data")
    @patch("web.web_interface_new.analyzer.run_analysis")
    @patch("web.web_interface_new.analyzer.extract_analysis_results")
    def test_analyze_updates_matching_holding(self, extract_results, run_analysis, fetch_market_data):
        fetch_market_data.return_value = __import__("pandas").DataFrame([
            {"Open": 100, "High": 110, "Low": 95, "Close": 125}
        ])
        run_analysis.return_value = {"success": True, "asset_name": "MSTR", "timeframe": "1d", "data_length": 1, "final_state": {}}
        extract_results.return_value = {
            "success": True,
            "asset_name": "MSTR",
            "timeframe": "1d",
            "data_length": 1,
            "technical_indicators": "",
            "pattern_analysis": "",
            "trend_analysis": "",
            "final_decision": {"decision": "BUY", "justification": "趋势仍强"},
            "single_name_score": {"decision": "BUY", "recommended_action": "加仓", "recommended_book": "核心仓"},
            "positions": [],
            "candidates": [],
            "account_state": {},
            "portfolio_directive": {},
            "dashboard_payload": {},
        }
        response = self.client.post("/api/analyze", json={
            "asset": "MSTR",
            "timeframe": "1d",
            "start_date": "2026-04-01",
            "end_date": "2026-04-28",
            "start_time": "00:00",
            "end_time": "23:59",
            "trading_strategy": "high_frequency",
            "generate_charts": False,
            "redirect_to_output": False,
        })
        payload = response.get_json()
        self.assertTrue(payload["workspace_writeback"]["updated"])
        workspace = db_manager.get_stage2_workspace("default")
        self.assertEqual(workspace["positions"][0]["latest_price"], 125.0)
        self.assertEqual(workspace["positions"][0]["unrealized_pnl"], 250.0)
```

- [ ] **Step 2: Run the write-back tests and verify they fail**

Run: `python -m unittest tests.test_stage2_writeback -v`
Expected: FAIL because `/api/analyze` does not yet return `workspace_writeback` or update the stored workspace.

- [ ] **Step 3: Add write-back metadata and apply it after successful single-name analysis**

```python
def extract_analysis_results(self, results: Dict[str, Any], latest_price: float | None = None) -> Dict[str, Any]:
    ...
    return {
        "success": True,
        "asset_name": safe_str(results["asset_name"]),
        "timeframe": safe_str(results["timeframe"]),
        "data_length": results["data_length"],
        "latest_price": latest_price,
        "analysis_summary": safe_str(final_decision.get("justification") or normalized_score.get("justification") or ""),
        ...
    }
```

```python
from services.stage2_workspace import refresh_position_after_analysis

latest_price = round(float(df.iloc[-1]["Close"]), 4)
formatted_results = analyzer.extract_analysis_results(results, latest_price=latest_price)
workspace_writeback = {"updated": False, "message": "未使用 Stage 2 工作区"}
workspace = db_manager.get_stage2_workspace("default")
if workspace:
    updated_workspace, workspace_writeback = refresh_position_after_analysis(
        workspace,
        asset=asset,
        latest_price=latest_price,
        analysis_summary=formatted_results.get("analysis_summary", ""),
    )
    if workspace_writeback["updated"]:
        db_manager.save_stage2_workspace(
            workspace_name=updated_workspace["workspace_name"],
            account_state=updated_workspace["account_state"],
            positions=updated_workspace["positions"],
            candidates=updated_workspace["candidates"],
            notes=updated_workspace.get("notes", ""),
        )
formatted_results["workspace_writeback"] = workspace_writeback
```

- [ ] **Step 4: Run the write-back tests and verify they pass**

Run: `python -m unittest tests.test_stage2_writeback -v`
Expected: `Ran 1 test` and `OK`

- [ ] **Step 5: Commit**

```bash
git add services/stage2_workspace.py web/web_interface_new.py tests/test_stage2_writeback.py
git commit -m "feat: write back stage2 holdings after analysis"
```

---

## Phase 2 — Account analysis workflow and history

### Task 5: Build the account-analysis service with artifact retention

**Files:**
- Create: `services/account_analysis.py`
- Create: `tests/test_account_analysis_service.py`

- [ ] **Step 1: Write the failing account-analysis service tests**

```python
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from services.account_analysis import save_account_analysis_artifacts, list_account_analysis_history


class AccountAnalysisServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_save_account_analysis_artifacts_keeps_only_latest_ten(self):
        for index in range(12):
            save_account_analysis_artifacts(
                output_dir=self.temp_dir,
                workspace_name="default",
                analysis_payload={"summary": f"run-{index}", "manager_actions": ["hold"]},
                markdown_body=f"# run-{index}\n",
                created_at=f"2026-04-28T18:{index:02d}:00",
            )
        history = list_account_analysis_history(self.temp_dir)
        self.assertEqual(len(history), 10)
        self.assertEqual(history[0]["summary"], "run-11")
        self.assertEqual(history[-1]["summary"], "run-2")
```

- [ ] **Step 2: Run the account-analysis service tests and verify they fail**

Run: `python -m unittest tests.test_account_analysis_service -v`
Expected: `ImportError: No module named 'services.account_analysis'`

- [ ] **Step 3: Implement artifact saving, retention, and history listing**

```python
import json
from pathlib import Path
from typing import Any, Dict, List


def save_account_analysis_artifacts(output_dir: Path, workspace_name: str, analysis_payload: Dict[str, Any], markdown_body: str, created_at: str) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_timestamp = created_at.replace(":", "-")
    stem = f"{safe_timestamp}_{workspace_name}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(analysis_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown_body, encoding="utf-8")
    json_files = sorted(output_dir.glob("*.json"), reverse=True)
    for stale_json in json_files[10:]:
        stale_md = stale_json.with_suffix('.md')
        stale_json.unlink(missing_ok=True)
        stale_md.unlink(missing_ok=True)
    return {"json_path": str(json_path), "markdown_path": str(md_path)}


def list_account_analysis_history(output_dir: Path) -> List[Dict[str, Any]]:
    if not output_dir.exists():
        return []
    history = []
    for json_path in sorted(output_dir.glob("*.json"), reverse=True):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        history.append({
            "created_at": payload.get("created_at"),
            "workspace_name": payload.get("workspace_name", "default"),
            "summary": payload.get("summary", ""),
            "json_path": str(json_path),
            "markdown_path": str(json_path.with_suffix('.md')),
        })
    return history[:10]
```

- [ ] **Step 4: Run the account-analysis service tests and verify they pass**

Run: `python -m unittest tests.test_account_analysis_service -v`
Expected: `Ran 1 test` and `OK`

- [ ] **Step 5: Commit**

```bash
git add services/account_analysis.py tests/test_account_analysis_service.py
git commit -m "feat: add account analysis artifact retention"
```

### Task 6: Expose account-analysis APIs and price refresh workflow

**Files:**
- Modify: `web/web_interface_new.py:598-653`
- Modify: `web/web_interface_new.py:1198-1495`
- Create: `tests/test_account_analysis_api.py`

- [ ] **Step 1: Write the failing account-analysis API tests**

```python
import unittest
from unittest.mock import patch

from web.web_interface_new import app, db_manager


class AccountAnalysisApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        db_manager.save_stage2_workspace(
            workspace_name="default",
            account_state={"nav": 100000, "cash": 25000, "current_drawdown": 6},
            positions=[{"ticker": "MSTR", "book_type": "核心仓", "cost_basis": 100, "shares": 10, "tracking_status": "持续跟踪", "factor_tags": ["btc-beta"]}],
            candidates=[{"ticker": "CRCL", "factor_tags": ["fintech"]}],
            notes="live book",
        )

    @patch("web.web_interface_new.analyzer.data_fetcher.fetch_yfinance_data_with_datetime")
    @patch("web.web_interface_new.analyzer.llm_provider.get_client")
    def test_account_analysis_returns_summary_and_history_entry(self, get_client, fetch_prices):
        fetch_prices.return_value = __import__("pandas").DataFrame([
            {"Datetime": "2026-04-28", "Open": 120, "High": 126, "Low": 119, "Close": 125, "Volume": 1000}
        ])
        fake_client = get_client.return_value
        fake_client.chat.completions.create.return_value.choices = [type("Choice", (), {"message": type("Msg", (), {"content": '{"summary":"组合健康","manager_actions":["减集中度"]}'})()})()]
        response = self.client.post("/api/account-analysis", json={"workspace_name": "default"})
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["analysis"]["summary"], "组合健康")
        history = self.client.get("/api/account-analysis/history").get_json()
        self.assertTrue(history["success"])
        self.assertGreaterEqual(len(history["history"]), 1)
```

- [ ] **Step 2: Run the account-analysis API tests and verify they fail**

Run: `python -m unittest tests.test_account_analysis_api -v`
Expected: FAIL because the routes do not exist.

- [ ] **Step 3: Implement the manual account-analysis route and history endpoints**

```python
from services.account_analysis import list_account_analysis_history, save_account_analysis_artifacts
from services.stage2_workspace import recalculate_account_state

ACCOUNT_ANALYSIS_DIR = Path(_project_root) / "artifacts" / "account_analysis"


@app.route('/api/account-analysis', methods=['POST'])
def run_account_analysis():
    data = request.get_json() or {}
    workspace_name = data.get('workspace_name', 'default')
    workspace = db_manager.get_stage2_workspace(workspace_name)
    if not workspace:
        return jsonify({"success": False, "error": "Workspace not found"}), 404
    workspace = recalculate_account_state(workspace)
    created_at = datetime.utcnow().isoformat(timespec='seconds')
    prompt = json.dumps({
        "account_state": workspace["account_state"],
        "positions": workspace["positions"],
        "candidates": workspace["candidates"],
        "output_schema": {
            "summary": "string",
            "portfolio_health_score": "number",
            "holding_health": "array",
            "pnl_breakdown": "object",
            "concentration_risks": "array",
            "crowded_exposures": "array",
            "manager_actions": "array"
        }
    }, ensure_ascii=False)
    client = analyzer.llm_provider.get_client()
    response = client.chat.completions.create(
        model=analyzer.llm_provider.providers[analyzer.llm_provider.current_provider]['models'][0],
        messages=[
            {"role": "system", "content": "You are a portfolio manager assistant. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    analysis = json.loads(response.choices[0].message.content)
    analysis["created_at"] = created_at
    analysis["workspace_name"] = workspace_name
    markdown_body = f"# Account Analysis\n\n- Summary: {analysis.get('summary', '')}\n"
    artifact_paths = save_account_analysis_artifacts(ACCOUNT_ANALYSIS_DIR, workspace_name, analysis, markdown_body, created_at)
    return jsonify({"success": True, "analysis": analysis, "artifacts": artifact_paths})


@app.route('/api/account-analysis/history', methods=['GET'])
def get_account_analysis_history():
    return jsonify({"success": True, "history": list_account_analysis_history(ACCOUNT_ANALYSIS_DIR)})
```

- [ ] **Step 4: Run the account-analysis API tests and verify they pass**

Run: `python -m unittest tests.test_account_analysis_api -v`
Expected: `Ran 1 test` and `OK`

- [ ] **Step 5: Commit**

```bash
git add web/web_interface_new.py tests/test_account_analysis_api.py
git commit -m "feat: add manual account analysis api"
```

### Task 7: Wire account-analysis controls and history into the existing page

**Files:**
- Modify: `templates/demo_new.html:1492-1594`
- Modify: `templates/demo_new.html:1683-1915`
- Modify: `templates/demo_new.html:3090-3204`
- Modify: `tests/test_demo_new_stage2_template.py`

- [ ] **Step 1: Extend the template regression test to require account-analysis fetch hooks**

```python
self.assertIn("async function runAccountAnalysis()", html)
self.assertIn("async function loadAccountAnalysisHistory()", html)
self.assertIn("workspaceLastAnalysisStatus", html)
```

- [ ] **Step 2: Run the template regression test and verify it fails**

Run: `python -m unittest tests.test_demo_new_stage2_template -v`
Expected: FAIL because the new account-analysis functions and status anchors are not present yet.

- [ ] **Step 3: Add frontend fetch/render logic for account analysis and history**

```javascript
async function runAccountAnalysis() {
    const messageEl = document.getElementById('stage2WorkspaceMessage');
    messageEl.textContent = 'Running account analysis...';
    messageEl.className = 'alert alert-info mt-3';
    messageEl.style.display = 'block';
    const response = await fetch('/api/account-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workspace_name: 'default' })
    });
    const data = await response.json();
    if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to run account analysis');
    }
    document.getElementById('workspaceLastAnalysisStatus').textContent = data.analysis.summary || 'Completed';
    messageEl.textContent = 'Account analysis completed.';
    messageEl.className = 'alert alert-success mt-3';
    await loadAccountAnalysisHistory();
}

async function loadAccountAnalysisHistory() {
    const response = await fetch('/api/account-analysis/history');
    const data = await response.json();
    const list = document.getElementById('workspaceAnalysisHistoryList');
    list.innerHTML = (data.history || []).map(item => `
        <div class="workspace-history-item">
            <strong>${escapeHtml(item.created_at || '')}</strong>
            <span>${escapeHtml(item.summary || '')}</span>
        </div>
    `).join('');
}
```

- [ ] **Step 4: Run the template regression test and verify it passes**

Run: `python -m unittest tests.test_demo_new_stage2_template -v`
Expected: `Ran 1 test` and `OK`

- [ ] **Step 5: Commit**

```bash
git add templates/demo_new.html tests/test_demo_new_stage2_template.py
git commit -m "feat: surface account analysis controls in stage2 ui"
```

---

## Final verification

### Task 8: Run the full verification pass

**Files:**
- Modify: none
- Test: `tests/test_stage2_workspace_service.py`
- Test: `tests/test_stage2_workspace_api.py`
- Test: `tests/test_stage2_writeback.py`
- Test: `tests/test_account_analysis_service.py`
- Test: `tests/test_account_analysis_api.py`
- Test: `tests/test_demo_new_stage2_template.py`

- [ ] **Step 1: Run the automated Phase 1 + Phase 2 test suite**

Run:

```bash
python -m unittest \
  tests.test_stage2_workspace_service \
  tests.test_stage2_workspace_api \
  tests.test_stage2_writeback \
  tests.test_account_analysis_service \
  tests.test_account_analysis_api \
  tests.test_demo_new_stage2_template -v
```

Expected: all tests pass with `OK`

- [ ] **Step 2: Run a syntax pass on the touched backend files**

Run:

```bash
python -m py_compile \
  services/stage2_workspace.py \
  services/account_analysis.py \
  services/database.py \
  web/web_interface_new.py
```

Expected: no output

- [ ] **Step 3: Start the app and perform the manual Phase 1 verification**

Run: `python run.py`
Expected: Flask app starts locally without import errors

Manual checks:

```text
1. Open the main page and verify the Analysis tab still loads.
2. Open Stage 2 Workspace and confirm the position editor shows Cost / Shares / Tracking / read-only derived columns.
3. Save a workspace, refresh the page, and confirm the rows reload from SQLite.
4. Run analysis for a held ticker and confirm the Stage 2 panel shows updated latest price / market value / PnL / weight.
5. Run analysis for a ticker not in holdings and confirm the workspace message says no write-back occurred.
```

- [ ] **Step 4: Perform the manual Phase 2 verification**

```text
1. Click Run Account Analysis.
2. Confirm the page shows a completed status and a recent history item.
3. Confirm JSON and Markdown files exist under artifacts/account_analysis/.
4. Run the account analysis repeatedly until more than 10 files exist, then confirm only the latest 10 remain.
5. Reload the page and confirm the history list still renders.
```

- [ ] **Step 5: Commit the verification-safe final state**

```bash
git add services/stage2_workspace.py services/account_analysis.py services/database.py web/web_interface_new.py templates/demo_new.html tests
git commit -m "feat: deliver stage2 account tracking and account analysis"
```

---

## Self-review checklist

- Spec coverage: Phase 1 covers frontend cleanup, richer position fields, derived metrics, and post-analysis write-back. Phase 2 covers manual account analysis, latest-price-driven account review, artifact retention, and recent history surfacing.
- Placeholder scan: no `TBD`, `TODO`, or “similar to above” instructions remain.
- Type consistency: `cost_basis`, `shares`, `tracking_status`, `latest_price`, `market_value`, `position_weight`, `unrealized_pnl`, and `unrealized_pnl_pct` are used consistently across service, API, and frontend tasks.
