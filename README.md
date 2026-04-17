# Data Science Agent Platform

A multi-stage, form-driven data science pipeline with an interactive Streamlit dashboard featuring advanced analytics, ML model training, and report generation.

## Features

### Data Cleaning
- Advanced imputation strategies (mean, median, mode, KNN, group-based)
- Z-score and IQR-based outlier detection
- Forward/backward fill for time series
- sklearn Pipeline wrapper

### Exploratory Data Analysis
- Statistical summaries with skewness interpretation
- Correlation heatmaps and pair analysis
- Violin plots and KDE distributions
- Interactive visualizations

### Feature Engineering
- Date expansion (year, month, day, weekday, quarter)
- Ratio and interaction features (products, powers, exponentials)
- Log transforms and aggregated features
- Custom binning

### Hypothesis Testing
- T-tests (independent, paired, one-sample)
- Chi-square tests for categorical data
- Automated statistical interpretation

### Machine Learning
- Automated model selection (Classifier/Regressor)
- ROC/AUC curves and precision/recall metrics
- Classification reports and confusion matrices

### Reporting
- Markdown report generation
- PDF export (via WeasyPrint)
- Technical, Executive, and General audience formats

## Tech Stack

- **UI:** Streamlit
- **Data:** Pandas, NumPy, Scikit-Learn, SciPy
- **Viz:** Seaborn, Matplotlib

## Project Structure

```
ds-agent-platform/
├── app.py                    # Streamlit dashboard
├── src/ds_platform/agents/   # Core agent logic
│   ├── cleaning.py           # Data cleaning
│   ├── eda.py               # EDA visualizations
│   ├── features.py          # Feature engineering
│   ├── model.py             # ML training
│   ├── report.py            # Report generation
│   └── stats.py             # Hypothesis testing
├── data/                     # Sample datasets
├── tests/                    # Unit tests
└── venv/                     # Virtual environment
```

## Quick Start

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

4. **Generate synthetic test data:**
   ```bash
   python data/generate_synthetic.py
   ```

## Testing

```bash
pytest tests/ -v
```

## License

MIT