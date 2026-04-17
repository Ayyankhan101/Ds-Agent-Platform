from pathlib import Path
from ds_platform.agents.report import ReportWriter


class TestReportWriter:
    def setup_method(self):
        self.output_dir = Path("reports")
        self.agent = ReportWriter(output_dir=str(self.output_dir))
        self.context = {
            "eda_results": {"summary": {"mean": 5}},
            "cleaning_logs": [{"task": "imputation"}],
            "features": [{"feature": "ratio", "type": "ratio"}],
            "stats_result": {"test_type": "t-test", "p_value": 0.05},
            "model_metrics": {"accuracy": 0.85},
        }

    def test_generate_report_creates_file(self):
        path = self.agent.generate_report("Test Report", "Technical", self.context)
        assert Path(path).exists()

    def test_report_contains_title(self):
        path = self.agent.generate_report("My Report", "Technical", self.context)
        with open(path) as f:
            content = f.read()
        assert "My Report" in content

    def test_report_contains_audience(self):
        path = self.agent.generate_report("Test", "Executive", self.context)
        with open(path) as f:
            content = f.read()
        assert "Executive" in content

    def test_report_contains_all_sections(self):
        path = self.agent.generate_report("Test", "General", self.context)
        with open(path) as f:
            content = f.read()
        assert "Executive Summary" in content
        assert "Data Cleaning" in content
        assert "Machine Learning" in content

    def test_report_handles_empty_context(self):
        path = self.agent.generate_report("Test", "Technical", {})
        assert Path(path).exists()

    def test_different_audiences(self):
        for audience in ["Technical", "Executive", "General"]:
            path = self.agent.generate_report("Test", audience, {})
            assert Path(path).exists()
            with open(path) as f:
                content = f.read()
            assert audience in content
