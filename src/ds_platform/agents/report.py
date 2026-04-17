import json
from pathlib import Path
from typing import Dict, Any


class ReportWriter:
    """
    Agent for aggregating all pipeline outputs into a final structured report.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, title: str, audience: str, context: Dict[str, Any]) -> str:
        report_path = self.output_dir / "final_report.md"

        sections = [
            f"# {title}",
            f"**Audience:** {audience}\n",
            "## 1. Executive Summary",
            "This report summarizes the data science pipeline execution.\n",
            "## 2. Data Cleaning & Transformation",
            f"Logs found: {len(context.get('cleaning_logs', []))} entries.\n",
            "## 3. Exploratory Data Analysis",
            "Key insights from correlation and distribution analysis.\n",
            "## 4. Feature Engineering",
            f"New features generated: {len(context.get('features', []))}\n",
            "## 5. Statistical Hypothesis Testing",
            f"Result: {context.get('stats_result', 'N/A')}\n",
            "## 6. Machine Learning Model Performance",
            f"Metrics: {json.dumps(context.get('model_metrics', {}), indent=2)}\n",
        ]

        content = "\n".join(sections)
        with open(report_path, "w") as f:
            f.write(content)

        return str(report_path)
