# Data Science Agent Platform

A multi-stage, form-driven data science pipeline powered by specialized agents.

## 🚀 Features

- **Task 1: Advanced Data Cleaning** - Imputation strategies and IQR-based outlier removal.
- **Task 2: Deep EDA** - Statistical summaries and correlation heatmaps.
- **Task 3: Feature Engineering** - Date expansion, ratio features, and binning.
- **Task 4: Hypothesis Testing** - t-tests and Chi-square tests with automated interpretation.
- **Task 5: Machine Learning** - Automated model selection (Classifier/Regressor) and evaluation.
- **Task 6: Insight Reporting** - Aggregated Markdown report generation.
- **Task 7: API Data Fetching** - Integration with external JSON APIs.

## 🛠️ Tech Stack

- **UI:** Streamlit
- **Data:** Pandas, NumPy, Scikit-Learn, SciPy
- **Viz:** Seaborn, Matplotlib
- **API:** Requests

## 📂 Project Structure

```
data-science-platform/
├── app.py              # Streamlit Entry Point
├── src/
│   └── ds_platform/
│       └── agents/     # Core Agent Logic
├── data/               # Output artifacts (CSV, JSON)
├── reports/            # Generated Markdown reports
├── notebooks/          # Exploratory work
└── tests/              # Unit tests
```

## 🚥 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

## 📄 License

MIT
