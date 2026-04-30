import re
import unittest
from pathlib import Path


class DemoNewStage2TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_path = Path(__file__).resolve().parents[1] / 'templates' / 'demo_new.html'
        cls.template_text = cls.template_path.read_text(encoding='utf-8')

    def test_template_contains_single_run_analysis_definition(self):
        self.assertEqual(self.template_text.count('function runAnalysis()'), 1)
        self.assertEqual(self.template_text.count('async function runAccountAnalysis()'), 1)

    def test_template_contains_stage2_workspace_markers(self):
        for marker in [
            'cost_basis',
            'tracking_status',
            'workspaceAccountAnalysisBtn',
            'workspaceAnalysisHistoryList',
            'workspaceLastAnalysisStatus',
            'Stage 2 Account Workspace',
            'Run Account Analysis',
            'workspaceCoreSkillPanel',
            'workspaceCoreSkillSummary',
            'workspaceCoreSkillActionsAdd',
            'workspaceCoreSkillActionsTrim',
            'workspaceCoreSkillActionsHold',
            'workspaceCoreSkillScoreTableBody',
            'function renderCoreSkillBlock(coreSkillBlock)',
            'async function runAccountAnalysis()',
            'async function loadAccountAnalysisHistory()',
            "fetch('/api/account-analysis',",
            "fetch('/api/account-analysis/history?workspace_name=default')",
            'loadAccountAnalysisHistory();',
        ]:
            with self.subTest(marker=marker):
                self.assertIn(marker, self.template_text)

    def test_template_uses_single_stage2_surface(self):
        self.assertEqual(self.template_text.count('id="stage2TabPanel"'), 1)
        self.assertNotIn('id="stage2WorkspacePanel"', self.template_text)
        self.assertEqual(self.template_text.count('function buildEmptyPosition()'), 1)

    def test_account_analysis_uses_single_column_layout_group(self):
        self.assertIn('class="panel-group stage2-panel-group"', self.template_text)
        self.assertIn('.stage2-panel-group {', self.template_text)
        self.assertIn('grid-template-columns: 1fr;', self.template_text)

    def test_template_renders_history_without_inner_html_interpolation(self):
        self.assertIn("document.createElement('li')", self.template_text)
        self.assertNotIn("${entry.summary || 'No summary provided.'}", self.template_text)


if __name__ == '__main__':
    unittest.main()
