import unittest
from pathlib import Path


class CustomAssetDeleteTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template_text = (Path(__file__).resolve().parents[1] / "templates" / "demo_new.html").read_text(encoding="utf-8")
        cls.web_text = (Path(__file__).resolve().parents[1] / "web" / "web_interface_new.py").read_text(encoding="utf-8")

    def test_custom_asset_buttons_include_delete_control(self):
        self.assertIn("custom-asset-wrapper", self.template_text)
        self.assertIn("removeCustomAsset", self.template_text)
        self.assertIn("/api/delete-custom-asset", self.template_text)
        self.assertIn("fa-times", self.template_text)

    def test_new_web_interface_exposes_delete_custom_asset_api(self):
        self.assertIn("/api/save-custom-asset", self.web_text)
        self.assertIn("/api/delete-custom-asset", self.web_text)
        self.assertIn("def delete_custom_asset", self.web_text)
        self.assertIn("delete_custom_asset", self.web_text)


if __name__ == "__main__":
    unittest.main()
