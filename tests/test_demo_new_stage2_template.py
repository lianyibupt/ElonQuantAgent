import unittest
from pathlib import Path


class DemoNewStage2TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_path = Path(__file__).resolve().parents[1] / 'templates' / 'demo_new.html'
        cls.template_text = cls.template_path.read_text(encoding='utf-8')

    def test_template_contains_single_run_analysis_definition(self):
        self.assertEqual(self.template_text.count('function runAnalysis()'), 1)

    def test_template_contains_stage2_workspace_markers(self):
        for marker in [
            'cost_basis',
            'tracking_status',
            'workspaceAccountAnalysisBtn',
            'workspaceAnalysisHistoryList',
        ]:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.template_text)


if __name__ == '__main__':
    unittest.main()
