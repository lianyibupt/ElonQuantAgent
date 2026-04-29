import json
import shutil
import tempfile
import unittest
from pathlib import Path

from services.account_analysis import (
    list_account_analysis_history,
    save_account_analysis_artifacts,
)


class AccountAnalysisServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "account-analysis"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_account_analysis_artifacts_writes_json_and_markdown_with_utf8(self):
        result = save_account_analysis_artifacts(
            output_dir=self.output_dir,
            workspace_name="主账户",
            analysis_payload={"summary": "组合稳定，关注英伟达"},
            markdown_body="# 分析\n保持耐心。",
            created_at="2026-04-28T10:15:00Z",
        )

        self.assertEqual(result["created_at"], "2026-04-28T10:15:00Z")
        self.assertEqual(result["workspace_name"], "主账户")
        self.assertEqual(result["summary"], "组合稳定，关注英伟达")
        self.assertTrue(result["json_path"].endswith("2026-04-28T10-15-00Z.json"))
        self.assertTrue(result["markdown_path"].endswith("2026-04-28T10-15-00Z.md"))

        with open(result["json_path"], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["created_at"], "2026-04-28T10:15:00Z")
        self.assertEqual(payload["workspace_name"], "主账户")
        self.assertEqual(payload["summary"], "组合稳定，关注英伟达")

        with open(result["json_path"], "r", encoding="utf-8") as handle:
            raw_json = handle.read()
        self.assertIn("主账户", raw_json)
        self.assertIn("英伟达", raw_json)

        with open(result["markdown_path"], "r", encoding="utf-8") as handle:
            markdown_body = handle.read()
        self.assertEqual(markdown_body, "# 分析\n保持耐心。")

    def test_save_account_analysis_artifacts_injects_required_fields_when_missing(self):
        result = save_account_analysis_artifacts(
            output_dir=self.output_dir,
            workspace_name="growth",
            analysis_payload={"positions": 5},
            markdown_body="body",
            created_at="2026-04-28T11:15:00Z",
        )

        with open(result["json_path"], "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.assertEqual(payload["created_at"], "2026-04-28T11:15:00Z")
        self.assertEqual(payload["workspace_name"], "growth")
        self.assertEqual(payload["positions"], 5)

    def test_list_account_analysis_history_returns_latest_ten_sorted_desc(self):
        for index in range(12):
            save_account_analysis_artifacts(
                output_dir=self.output_dir,
                workspace_name=f"ws-{index}",
                analysis_payload={"summary": f"summary-{index}"},
                markdown_body=f"body-{index}",
                created_at=f"2026-04-28T10:{index:02d}:00Z",
            )

        history = list_account_analysis_history(self.output_dir)

        self.assertEqual(len(history), 10)
        self.assertEqual(history[0]["created_at"], "2026-04-28T10:11:00Z")
        self.assertEqual(history[-1]["created_at"], "2026-04-28T10:02:00Z")
        self.assertEqual(history[0]["workspace_name"], "ws-11")
        self.assertEqual(history[0]["summary"], "summary-11")

        remaining_json_files = sorted(path.name for path in self.output_dir.glob("*.json"))
        self.assertEqual(len(remaining_json_files), 10)
        self.assertNotIn("2026-04-28T10-00-00Z.json", remaining_json_files)
        self.assertNotIn("2026-04-28T10-01-00Z.json", remaining_json_files)

    def test_list_account_analysis_history_skips_missing_directory_and_bad_json(self):
        missing_history = list_account_analysis_history(self.output_dir)
        self.assertEqual(missing_history, [])

        self.output_dir.mkdir(parents=True, exist_ok=True)
        bad_json_path = self.output_dir / "2026-04-28T10-15-00Z.json"
        bad_json_path.write_text("{not-json", encoding="utf-8")

        partial_json_path = self.output_dir / "2026-04-28T10-16-00Z.json"
        partial_json_path.write_text(json.dumps({"summary": "fallback summary"}, ensure_ascii=False), encoding="utf-8")

        history = list_account_analysis_history(self.output_dir)

        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["summary"], "fallback summary")
        self.assertEqual(history[0]["workspace_name"], "")
        self.assertEqual(history[0]["created_at"], "2026-04-28T10-16-00Z")
        self.assertTrue(history[0]["markdown_path"].endswith("2026-04-28T10-16-00Z.md"))


if __name__ == "__main__":
    unittest.main()
