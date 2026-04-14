import streamlit as st
import pandas as pd
import json
from ds_platform.agents.cleaning import CleaningPipeline
from ds_platform.agents.eda import EDAAgent
from ds_platform.agents.features import FeatureEngineer
from ds_platform.agents.stats import StatsAgent
from ds_platform.agents.model import ModelTrainer
from ds_platform.agents.report import ReportWriter
from ds_platform.agents.api import APIFetcher

st.set_page_config(page_title="Data Science Agent Platform", layout="wide")

st.title("🚀 Data Science Agent Platform")
st.sidebar.title("Pipeline Steps")

# Initialize Agents
cleaning_agent = CleaningPipeline()
eda_agent = EDAAgent()
feature_agent = FeatureEngineer()
stats_agent = StatsAgent()
model_agent = ModelTrainer()
report_agent = ReportWriter()
api_agent = APIFetcher()

# Session State for Data Flow
if 'df' not in st.session_state:
    st.session_state.df = None
if 'context' not in st.session_state:
    st.session_state.context = {}

# Sidebar Navigation
step = st.sidebar.radio("Go to", [
    "1. Data Cleaning", 
    "2. Exploratory Data Analysis", 
    "3. Feature Engineering", 
    "4. Hypothesis Testing", 
    "5. Machine Learning", 
    "6. Insight Reporting",
    "7. API Data Fetching"
])

# --- Task 1: Data Cleaning ---
if step == "1. Data Cleaning":
    st.header("🧹 Task 1: Advanced Data Cleaning")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "json"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.write("### Raw Data Preview", df.head())
        
        cols = df.columns.tolist()
        strategy = {}
        st.subheader("Imputation Strategy")
        for col in df.select_dtypes(include=['number']).columns:
            strategy[col] = st.selectbox(f"Strategy for {col}", ["mean", "median", "mode", "none"], key=col)
            
        st.subheader("Outlier Config")
        outlier_method = st.toggle("Enable IQR Outlier Removal", value=True)
        threshold = st.slider("IQR Threshold", 1.0, 3.0, 1.5)
        
        if st.button("Run Cleaning Pipeline"):
            clean_df = cleaning_agent.clean(df, strategy, {"method": "IQR" if outlier_method else None, "threshold": threshold})
            st.session_state.df = clean_df
            st.success("Data cleaned and saved to cleaned_data.csv")
            st.write("### Cleaned Data Preview", clean_df.head())

# --- Task 2: EDA ---
elif step == "2. Exploratory Data Analysis":
    st.header("📊 Task 2: Deep EDA")
    if st.session_state.df is not None:
        target = st.selectbox("Select Target Variable", st.session_state.df.columns)
        method = st.radio("Correlation Method", ["pearson", "spearman"])
        
        if st.button("Run Analysis"):
            results = eda_agent.analyze(st.session_state.df, target, method)
            st.session_state.context['eda_results'] = results
            st.write("### Summary Statistics", results['summary'])
            st.pyplot(eda_agent.plot_correlations(st.session_state.df))
    else:
        st.warning("Please run Task 1 first.")

# --- Task 3: Feature Engineering ---
elif step == "3. Feature Engineering":
    st.header("⚙️ Task 3: Feature Engineering")
    if st.session_state.df is not None:
        date_cols = st.multiselect("Select Date Columns", st.session_state.df.columns)
        # Simplified ratio input for demo
        num_col = st.selectbox("Numerator Column", st.session_state.df.select_dtypes(include='number').columns)
        den_col = st.selectbox("Denominator Column", st.session_state.df.select_dtypes(include='number').columns)
        
        if st.button("Engineer Features"):
            config = {
                "date_columns": date_cols,
                "ratio_pairs": [(num_col, den_col)]
            }
            feat_df = feature_agent.transform(st.session_state.df, config)
            st.session_state.df = feat_df
            st.success("Features created!")
            st.write(feat_df.head())
    else:
        st.warning("Please load data in Task 1.")

# --- Task 4: Hypothesis Testing ---
elif step == "4. Hypothesis Testing":
    st.header("🧪 Task 4: Hypothesis Testing")
    if st.session_state.df is not None:
        h0 = st.text_input("H0 (Null Hypothesis)")
        h1 = st.text_input("H1 (Alternative Hypothesis)")
        test_type = st.selectbox("Test Type", ["t-test", "chi-square"])
        
        col1 = st.selectbox("Select Column", st.session_state.df.columns)
        
        if st.button("Run Test"):
            res = stats_agent.run_test(st.session_state.df, test_type, {"column": col1})
            st.session_state.context['stats_result'] = res
            st.write(res)
    else:
        st.warning("Please load data first.")

# --- Task 5: Machine Learning ---
elif step == "5. Machine Learning":
    st.header("🤖 Task 5: Machine Learning")
    if st.session_state.df is not None:
        target = st.selectbox("Select Target", st.session_state.df.columns)
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        seed = st.number_input("Random Seed", value=42)
        
        if st.button("Train Model"):
            res = model_agent.train_and_evaluate(st.session_state.df, target, {"test_size": test_size, "random_seed": seed})
            st.session_state.context['model_metrics'] = res['metrics']
            st.write("### Model Results", res)
    else:
        st.warning("Please load data first.")

# --- Task 6: Reporting ---
elif step == "6. Insight Reporting":
    st.header("📝 Task 6: Insight Reporting")
    title = st.text_input("Report Title", "Data Science Project Report")
    audience = st.selectbox("Audience", ["Technical", "Executive", "General"])
    
    if st.button("Generate Final Report"):
        path = report_agent.generate_report(title, audience, st.session_state.context)
        st.success(f"Report generated at {path}")
        with open(path, "r") as f:
            st.markdown(f.read())

# --- Task 7: API Fetching ---
elif step == "7. API Data Fetching":
    st.header("🌐 Task 7: API Data Fetching")
    url = st.text_input("API Endpoint URL")
    api_key = st.text_input("API Key (if required)", type="password")
    
    if st.button("Fetch Data"):
        api_df = api_agent.fetch(url, headers={"Authorization": f"Bearer {api_key}"} if api_key else None)
        if not api_df.empty:
            st.write("### Fetched Data", api_df.head())
            if st.button("Load into Pipeline"):
                st.session_state.df = api_df
                st.success("API data loaded into session!")
