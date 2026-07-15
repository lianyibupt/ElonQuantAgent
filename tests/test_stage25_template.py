import unittest
from pathlib import Path


class Stage25TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = (Path(__file__).resolve().parents[1] / "templates" / "demo_new.html").read_text(encoding="utf-8")
        cls.output = (Path(__file__).resolve().parents[1] / "templates" / "output.html").read_text(encoding="utf-8")
        cls.web = (Path(__file__).resolve().parents[1] / "web" / "web_interface_new.py").read_text(encoding="utf-8")

    def test_workspace_table_contains_stage25_plan_fields(self):
        for marker in [
            "stop_price",
            "target_price",
            "planned_action",
            "risk_to_stop",
            "risk_to_stop_pct_nav",
            "risk_reward_to_plan",
        ]:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.demo)

    def test_output_and_api_include_action_queue(self):
        self.assertIn("action_queue", self.web)
        self.assertIn("Action Queue", self.output)
        self.assertIn("total_risk_to_stop_pct_nav", self.output)


if __name__ == "__main__":
    unittest.main()
