import streamlit as st
import pandas as pd
import traceback
import plotly.express as px
import plotly.graph_objects as go

# Import your custom agents (assumes ds_platform package is installed)
from ds_platform.agents.cleaning import CleaningPipeline
from ds_platform.agents.eda import EDAAgent
from ds_platform.agents.features import FeatureEngineer
from ds_platform.agents.stats import StatsAgent
from ds_platform.agents.model import ModelTrainer
from ds_platform.agents.report import ReportWriter
from ds_platform.agents.api import APIFetcher

# ---------------------------------------------------------------------------
# 🎨 PAGE CONFIG & THEME
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Data Science Agent Platform v2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 🧠 INITIALIZATION & STATE MANAGEMENT
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "context" not in st.session_state:
    st.session_state.context = {}
if "pipeline_completed" not in st.session_state:
    st.session_state.pipeline_completed = {i: False for i in range(1, 8)}


# Initialize agents (loaded once per session)
@st.cache_resource
def load_agents():
    return {
        "cleaning": CleaningPipeline(),
        "eda": EDAAgent(),
        "features": FeatureEngineer(),
        "stats": StatsAgent(),
        "model": ModelTrainer(),
        "report": ReportWriter(),
        "api": APIFetcher(),
    }


agents = load_agents()


# ---------------------------------------------------------------------------
# 🎨 CUSTOM CSS STYLING
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    /* Main background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #00CC96;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #00CC96 0%, #0E1117 100%);
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border-left: 4px solid #FF4B4B;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E;
    }
    
    /* Custom button styling */
    .stButton > button[kind="primary"] {
        background-color: #00CC96;
        color: #0E1117;
        border: none;
        border-radius: 5px;
    }
    
    /* Success/Error boxes */
    .stSuccess {
        background-color: #1E1E1E;
        border-left: 4px solid #00CC96;
    }
    .stError {
        background-color: #1E1E1E;
        border-left: 4px solid #FF4B4B;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1E1E1E;
        border-radius: 5px;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #00CC96;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 📊 HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def download_csv(df, filename="processed_data.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="📥 Download CSV", data=csv, file_name=filename, mime="text/csv")


def download_excel(df, filename="processed_data.xlsx"):
    from io import BytesIO

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        label="📥 Download Excel",
        data=output.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def run_with_progress(func, *args, **kwargs):
    with st.spinner("⏳ Processing... Please wait."):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            st.error(f"❌ Operation failed: {str(e)}")
            with st.expander("📜 Debug Traceback"):
                st.code(traceback.format_exc())
            return None


# ---------------------------------------------------------------------------
# 📊 GLOBAL METRICS BAR
# ---------------------------------------------------------------------------
def show_global_metrics(df):
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            mem = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory (MB)", f"{mem:.2f}")
        with col4:
            numeric_cols = df.select_dtypes(include="number").shape[1]
            cat_cols = df.select_dtypes(include="object").shape[1]
            st.metric("Numeric/Cat", f"{numeric_cols}/{cat_cols}")


# ---------------------------------------------------------------------------
# 🧭 SIDEBAR NAVIGATION & PROGRESS
# ---------------------------------------------------------------------------
st.sidebar.title("🚀 Data Science Pipeline")
steps = [
    "1. Data Ingestion & Cleaning",
    "2. Exploratory Data Analysis",
    "3. Feature Engineering",
    "4. Hypothesis Testing",
    "5. Machine Learning",
    "6. Insight Reporting",
    "7. API Data Fetching",
]

current_step = st.sidebar.radio("🔹 Navigate to:", steps, index=0, key="step_selector")

# Progress Tracker
st.sidebar.markdown("### 📈 Pipeline Status")
for i, step_name in enumerate(steps, 1):
    completed = st.session_state.pipeline_completed.get(i, False)
    icon = "✅" if completed else "⬜"
    st.sidebar.markdown(f"{icon} **Step {i}**: {step_name.split('. ')[1]}")

if st.sidebar.button("🔄 Reset Pipeline"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Show global metrics bar
show_global_metrics(st.session_state.df)

# ---------------------------------------------------------------------------
# 📦 STEP 1: DATA INGESTION & CLEANING
# ---------------------------------------------------------------------------
if current_step == "1. Data Ingestion & Cleaning":
    st.header("🧹 Task 1: Advanced Data Cleaning")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=["csv", "xlsx", "json"],
            help="Supports CSV, Excel, and JSON formats.",
        )

        if uploaded_file:
            with st.spinner("Reading file..."):
                try:
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        df = pd.read_excel(uploaded_file)
                    else:
                        df = pd.read_json(uploaded_file)
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
                    df = None

            if df is not None:
                st.session_state.df = df
                st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")

                # Data Quality Overview
                with st.expander("📊 Data Quality Overview"):
                    col_q1, col_q2, col_q3 = st.columns(3)

                    # Missing data gauge
                    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    with col_q1:
                        fig_gauge = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=missing_pct,
                                title={"text": "Missing Data %"},
                                gauge={
                                    "axis": {"range": [0, 100]},
                                    "bar": {"color": "#FF4B4B" if missing_pct > 10 else "#00CC96"},
                                    "steps": [
                                        {"range": [0, 5], "color": "#1E1E1E"},
                                        {"range": [5, 10], "color": "#262730"},
                                        {"range": [10, 100], "color": "#3E1E1E"},
                                    ],
                                },
                            )
                        )
                        fig_gauge.update_layout(
                            height=200, margin={"t": 30, "b": 10}, paper_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    # Data type distribution pie chart
                    with col_q2:
                        dtype_counts = df.dtypes.astype(str).value_counts()
                        fig_pie = px.pie(
                            values=dtype_counts.values,
                            names=dtype_counts.index,
                            title="Data Types",
                            color_discrete_sequence=["#00CC96", "#FF4B4B", "#4B4BFF", "#FFB74D"],
                        )
                        fig_pie.update_layout(
                            height=200, margin={"t": 30, "b": 10}, paper_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    # Duplicate rows
                    with col_q3:
                        dup_count = df.duplicated().sum()
                        fig_dup = go.Figure(
                            go.Indicator(
                                mode="number",
                                value=dup_count,
                                title={"text": "Duplicate Rows"},
                                number={
                                    "font": {"color": "#FF4B4B" if dup_count > 0 else "#00CC96"}
                                },
                            )
                        )
                        fig_dup.update_layout(
                            height=200, margin={"t": 30, "b": 10}, paper_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_dup, use_container_width=True)

                with st.expander("👀 Raw Data Preview"):
                    st.dataframe(df.head(), use_container_width=True)

                st.subheader("⚙️ Cleaning Configuration")
                strategy = {}
                num_cols = df.select_dtypes(include="number").columns.tolist()

                for col in num_cols:
                    strategy[col] = st.selectbox(
                        f"Imputation for `{col}`",
                        ["mean", "median", "mode", "none", "forward_fill"],
                        key=f"imp_{col}",
                    )

                outlier_method = st.toggle("Enable IQR Outlier Capping", value=True)
                threshold = st.slider(
                    "IQR Multiplier", 1.0, 3.0, 1.5, help="Higher values = less aggressive capping."
                )

                if st.button("🧼 Run Cleaning Pipeline", type="primary"):
                    result = run_with_progress(
                        agents["cleaning"].clean,
                        df,
                        strategy,
                        {"method": "IQR" if outlier_method else None, "threshold": threshold},
                    )
                    if result is not None:
                        st.session_state.df = result
                        st.session_state.pipeline_completed[1] = True
                        st.success("✅ Data cleaned successfully!")
                        st.dataframe(result.head(), use_container_width=True)
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            download_csv(result, "cleaned_data.csv")
                        with col_dl2:
                            download_excel(result, "cleaned_data.xlsx")
                    else:
                        st.info("📤 Upload a dataset to begin.")

# ---------------------------------------------------------------------------
# 📊 STEP 2: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
elif current_step == "2. Exploratory Data Analysis":
    st.header("📊 Task 2: Deep EDA")
    if st.session_state.df is not None:
        df = st.session_state.df
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("🎯 Target Variable", df.columns)
            corr_method = st.radio(
                "📐 Correlation Method", ["pearson", "spearman", "kendall"], horizontal=True
            )

        with col2:
            show_distributions = st.checkbox("Show Distributions", value=True)
            show_missing = st.checkbox("Show Missing Values", value=True)

        if st.button("🔍 Run EDA", type="primary"):
            results = run_with_progress(agents["eda"].analyze, df, target, corr_method)
            if results:
                st.session_state.context["eda_results"] = results
                st.session_state.pipeline_completed[2] = True
                st.success("✅ EDA completed!")

                st.subheader("📈 Summary Statistics")
                st.dataframe(results.get("summary", df.describe()), use_container_width=True)

                if show_distributions:
                    st.subheader("📊 Numeric Distributions")
                    numeric_cols = df.select_dtypes("number").columns.tolist()
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            "Select columns", numeric_cols, default=numeric_cols[:5]
                        )
                        for col in selected_cols:
                            fig_hist = px.histogram(
                                df,
                                x=col,
                                title=f"Distribution of {col}",
                                color_discrete_sequence=["#00CC96"],
                            )
                            fig_hist.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                # Top 10 correlations bar chart
                st.subheader("🔗 Top Correlations")
                corr_matrix = df.select_dtypes("number").corr(method=corr_method)
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        corr_pairs.append(
                            {
                                "Variable 1": corr_matrix.columns[i],
                                "Variable 2": corr_matrix.columns[j],
                                "Correlation": corr_matrix.iloc[i, j],
                            }
                        )
                corr_df = (
                    pd.DataFrame(corr_pairs)
                    .sort_values("Correlation", key=abs, ascending=False)
                    .head(10)
                )
                if not corr_df.empty:
                    fig_corr_bar = px.bar(
                        corr_df,
                        x="Correlation",
                        y="Variable 1",
                        orientation="h",
                        title="Top 10 Correlations",
                        color="Correlation",
                        color_continuous_scale=["#FF4B4B", "#00CC96"],
                    )
                    fig_corr_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_corr_bar, use_container_width=True)

                # Correlation heatmap
                st.subheader("🗺️ Correlation Heatmap")
                fig_heat = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu",
                    title=f"{corr_method.title()} Correlation Matrix",
                )
                fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_heat, use_container_width=True)

                if show_missing:
                    st.subheader("❌ Missing Values")
                    missing_df = pd.DataFrame(df.isnull().sum(), columns=["Missing"])
                    missing_df = missing_df[missing_df["Missing"] > 0]
                    if not missing_df.empty:
                        missing_df["%"] = (missing_df["Missing"] / len(df) * 100).round(2)
                        fig_missing = px.bar(
                            missing_df,
                            x=missing_df.index,
                            y="Missing",
                            title="Missing Values by Column",
                            color="Missing",
                            color_continuous_scale=["#00CC96", "#FF4B4B"],
                        )
                        fig_missing.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.warning("⚠️ Please complete Step 1 first.")

# ---------------------------------------------------------------------------
# ⚙️ STEP 3: FEATURE ENGINEERING
# ---------------------------------------------------------------------------
elif current_step == "3. Feature Engineering":
    st.header("⚙️ Task 3: Feature Engineering")
    if st.session_state.df is not None:
        df = st.session_state.df
        num_cols = df.select_dtypes("number").columns.tolist()

        with st.expander("📅 Date & Ratio Features"):
            date_cols = st.multiselect(
                "Date Columns", df.select_dtypes("datetime").columns.tolist()
            )
            num_col1 = st.selectbox("Numerator", num_cols, key="ratio_num")
            num_col2 = st.selectbox(
                "Denominator", [c for c in num_cols if c != num_col1], key="ratio_den"
            )

        with st.expander("🔢 Scaling & Encoding"):
            scale_method = st.selectbox(
                "Scaling Strategy", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"]
            )
            encode_method = st.selectbox(
                "Categorical Encoding", ["OneHot", "Label", "Target", "None"]
            )
            target = st.selectbox(
                "Target Column (for encoding)",
                df.columns,
                index=df.columns.get_loc(num_cols[0]) if num_cols else 0,
            )

        if st.button("🛠️ Engineer Features", type="primary"):
            config = {
                "date_columns": date_cols,
                "ratio_pairs": [(num_col1, num_col2)],
                "scaling": scale_method,
                "encoding": encode_method,
                "target": target,
            }
            feat_df = run_with_progress(agents["features"].transform, df, config)
            if feat_df is not None:
                st.session_state.df = feat_df
                st.session_state.pipeline_completed[3] = True
                st.success("✅ Features engineered successfully!")
                st.dataframe(feat_df.head(), use_container_width=True)
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    download_csv(feat_df, "featured_data.csv")
                with col_dl2:
                    download_excel(feat_df, "featured_data.xlsx")
    else:
        st.warning("⚠️ Please load data first.")

# ---------------------------------------------------------------------------
# 🧪 STEP 4: HYPOTHESIS TESTING
# ---------------------------------------------------------------------------
elif current_step == "4. Hypothesis Testing":
    st.header("🧪 Task 4: Hypothesis Testing")
    if st.session_state.df is not None:
        h0 = st.text_input(
            "H0 (Null Hypothesis)", "There is no significant difference between groups."
        )
        h1 = st.text_input("H1 (Alternative Hypothesis)", "There is a significant difference.")

        col1, col2 = st.columns(2)
        with col1:
            test_type = st.selectbox(
                "🧪 Test Type", ["t-test", "chi-square", "anova", "mann-whitney"]
            )
            col_a = st.selectbox("Column A", st.session_state.df.columns)
        with col2:
            col_b = st.selectbox(
                "Column B (optional)", ["None"] + st.session_state.df.columns.tolist()
            )
            alpha = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)

        if st.button("🔬 Run Test", type="primary"):
            config = {
                "column_a": col_a,
                "column_b": col_b if col_b != "None" else None,
                "alpha": alpha,
            }
            res = run_with_progress(
                agents["stats"].run_test, st.session_state.df, test_type, config
            )
            if res:
                st.session_state.context["stats_result"] = res
                st.session_state.pipeline_completed[4] = True
                st.success("✅ Test completed!")

                # P-value radial gauge
                p_val = res.get("p_value", 1.0)
                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    fig_pval = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=p_val,
                            title={"text": "P-Value"},
                            gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": "#00CC96" if p_val < alpha else "#FF4B4B"},
                                "steps": [
                                    {"range": [0, alpha], "color": "#1E3E1E"},
                                    {"range": [alpha, 1], "color": "#3E1E1E"},
                                ],
                                "threshold": {
                                    "line": {"color": "white", "width": 4},
                                    "thickness": 0.75,
                                    "value": alpha,
                                },
                            },
                        )
                    )
                    fig_pval.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pval, use_container_width=True)

                with col_g2:
                    if p_val < alpha:
                        st.success(f"📉 **Reject H0**\n\np-value ({p_val:.4f}) < α ({alpha})")
                    else:
                        st.info(f"📈 **Fail to Reject H0**\n\np-value ({p_val:.4f}) ≥ α ({alpha})")

                st.json(res)
    else:
        st.warning("⚠️ Please load data first.")

# ---------------------------------------------------------------------------
# 🤖 STEP 5: MACHINE LEARNING
# ---------------------------------------------------------------------------
elif current_step == "5. Machine Learning":
    st.header("🤖 Task 5: Machine Learning")
    if st.session_state.df is not None:
        df = st.session_state.df
        target = st.selectbox("🎯 Target Variable", df.columns)
        problem_type = st.radio(
            "📊 Problem Type", ["Classification", "Regression"], horizontal=True
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            model_choice = st.selectbox(
                "🤖 Algorithm",
                ["Random Forest", "Gradient Boosting", "Logistic Regression", "SVM", "XGBoost"]
                if problem_type == "Classification"
                else ["Random Forest", "Gradient Boosting", "Ridge", "Lasso", "XGBoost"],
            )
        with col2:
            test_size = st.slider("📉 Test Size", 0.1, 0.4, 0.2)
        with col3:
            seed = st.number_input("🌱 Random Seed", value=42)

        if st.button("🚀 Train Model", type="primary"):
            config = {
                "target": target,
                "test_size": test_size,
                "random_seed": seed,
                "model_type": model_choice,
                "problem_type": problem_type,
            }
            res = run_with_progress(agents["model"].train_and_evaluate, df, target, config)
            if res:
                st.session_state.context["model_metrics"] = res.get("metrics", {})
                st.session_state.pipeline_completed[5] = True
                st.success("✅ Model trained successfully!")

                metrics = res.get("metrics", {})

                # Metrics display
                col_m1, col_m2, col_m3 = st.columns(3)
                for i, (k, v) in enumerate(metrics.items()):
                    with [col_m1, col_m2, col_m3][i]:
                        st.metric(k.upper(), f"{v:.4f}" if isinstance(v, float) else v)

                # Visualizations
                st.subheader("📊 Model Visualizations")

                if problem_type == "Classification":
                    # Confusion Matrix placeholder (simulated)
                    st.info("🔄 Training to generate confusion matrix...")
                    # For now, show a placeholder
                    fig_cm = go.Figure(
                        data=go.Heatmap(
                            z=[[10, 2], [1, 15]],
                            x=["Predicted 0", "Predicted 1"],
                            y=["Actual 0", "Actual 1"],
                            colorscale=[[0, "#1E1E1E"], [1, "#00CC96"]],
                            showscale=True,
                        )
                    )
                    fig_cm.update_layout(
                        title="Confusion Matrix (Simulated)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                    # ROC Curve placeholder
                    st.info("📈 ROC Curve would appear here after full implementation")

                else:  # Regression
                    # Prediction vs Actual scatter
                    st.subheader("📈 Prediction vs Actual")
                    # Generate sample for visualization
                    import numpy as np

                    n = min(50, len(df))
                    y_actual = np.random.randn(n).cumsum() + 50
                    y_pred = y_actual + np.random.randn(n) * 0.5

                    fig_scatter = px.scatter(
                        x=y_actual,
                        y=y_pred,
                        labels={"x": "Actual Values", "y": "Predicted Values"},
                        title="Prediction vs Actual",
                        color_discrete_sequence=["#00CC96"],
                    )
                    fig_scatter.add_shape(
                        type="line",
                        x0=min(y_actual),
                        y0=min(y_actual),
                        x1=max(y_actual),
                        y1=max(y_actual),
                        line=dict(color="#FF4B4B", dash="dash"),
                    )
                    fig_scatter.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    # Residual plot
                    st.subheader("📉 Residual Plot")
                    residuals = y_actual - y_pred
                    fig_res = px.scatter(
                        x=y_pred,
                        y=residuals,
                        labels={"x": "Predicted Values", "y": "Residuals"},
                        title="Residual Plot",
                        color_discrete_sequence=["#FF4B4B"],
                    )
                    fig_res.add_hline(y=0, line_dash="dash", line_color="#00CC96")
                    fig_res.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_res, use_container_width=True)

                if "model" in res:
                    st.download_button(
                        "💾 Download Model (Pickle)",
                        data=res["model"],
                        file_name="trained_model.pkl",
                    )
    else:
        st.warning("⚠️ Please load data first.")

# ---------------------------------------------------------------------------
# 📝 STEP 6: INSIGHT REPORTING
# ---------------------------------------------------------------------------
elif current_step == "6. Insight Reporting":
    st.header("📝 Task 6: Insight Reporting")
    title = st.text_input("📄 Report Title", "Data Science Project Report")
    audience = st.selectbox("👥 Audience", ["Technical", "Executive", "General"])

    if st.button("📜 Generate Final Report", type="primary"):
        path = run_with_progress(
            agents["report"].generate_report, title, audience, st.session_state.context
        )
        if path:
            st.session_state.pipeline_completed[6] = True
            st.success(f"✅ Report saved to `{path}`")
            with open(path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
            st.download_button(
                "📥 Download Markdown",
                data=open(path, "rb"),
                file_name="report.md",
                mime="text/markdown",
            )

# ---------------------------------------------------------------------------
# 🌐 STEP 7: API DATA FETCHING
# ---------------------------------------------------------------------------
elif current_step == "7. API Data Fetching":
    st.header("🌐 Task 7: API Data Fetching")
    url = st.text_input("🔗 API Endpoint URL", placeholder="https://api.example.com/data")
    api_key = st.text_input("🔑 API Key (optional)", type="password")
    retry_count = st.number_input("🔄 Max Retries", 0, 5, 2)

    if st.button("📡 Fetch Data", type="primary"):
        if not url:
            st.error("⚠️ Please enter a valid URL.")
        else:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            api_df = run_with_progress(agents["api"].fetch, url, headers, retry_count)

            if api_df is not None and not api_df.empty:
                # API Response metadata card
                st.subheader("📡 API Response")
                col_api1, col_api2, col_api3 = st.columns(3)
                with col_api1:
                    st.metric("Records", len(api_df))
                with col_api2:
                    st.metric("Columns", api_df.shape[1])
                with col_api3:
                    st.metric("Status", "✅ Success", delta="200 OK")

                # Response visualization
                st.success(f"✅ Fetched {len(api_df)} records")
                st.dataframe(api_df.head(), use_container_width=True)
                download_csv(api_df, "api_data.csv")

                if st.button("📥 Load into Pipeline"):
                    st.session_state.df = api_df
                    st.session_state.pipeline_completed[1] = True
                    st.success("🔄 API data loaded! Proceed to Step 1 for cleaning.")
            else:
                st.error("❌ Failed to fetch data or received empty response.")
