"""
Data Science Agent Platform - Full Dashboard Restyle
Matching reference_app.py MediTrack style with dark teal/neon theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Import DS Platform agents
from ds_platform.agents.cleaning import CleaningPipeline, SklearnPipelineWrapper
from ds_platform.agents.eda import EDAAgent
from ds_platform.agents.features import FeatureEngineer
from ds_platform.agents.stats import StatsAgent
from ds_platform.agents.model import ModelTrainer
from ds_platform.agents.report import ReportWriter
from ds_platform.agents.api import APIFetcher

# ============================================================================
# 🔧 PAGE CONFIG & THEME
# ============================================================================
st.set_page_config(
    page_title="Data Science Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# 🎨 DARK COMMAND CENTER THEME (matching reference_app.py)
# ============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root{
    --bg-dark:#0a0f0d; --bg-card:#0d1411; --bg-card-hover:#112420;
    --teal:#0d9488; --teal-light:#ccfbf1; --teal-dark:#134e4a;
    --teal-neon:#14ffec; --teal-dim:#0a6b68;
    --red:#ef4444; --amber:#f59e0b; --sky:#0ea5e9;
    --text:#e2e8f0; --text-muted:#94a3b8; --text-dim:#64748b;
    --border:#1e3a35; --glow:rgba(20,255,236,0.3);
}

html,body,[class*="css"]{background:var(--bg-dark)!important;color:var(--text)!important;}
h1,h2,h3{font-family:'DM Serif Display',serif;color:var(--teal-light)!important;}
p,div,span,label{font-family:'DM Sans',sans-serif;}
.stApp{background:var(--bg-dark)!important;}

/* Sidebar */
[data-testid="stSidebar"]{background:#0d1411!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--teal-light)!important;}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2{color:var(--teal-neon)!important;}

/* Metric Cards */
.metric-card{
    background:linear-gradient(135deg,var(--bg-card) 0%,#0f1f1c 100%);
    border:1px solid var(--border);border-radius:16px;
    padding:20px 24px;box-shadow:0 4px 20px rgba(0,0,0,0.4),0 0 30px var(--glow);
    transition:transform 0.2s ease,box-shadow 0.2s ease;
}
.metric-card:hover{transform:translateY(-2px);box-shadow:0 8px 30px rgba(0,0,0,0.5),0 0 40px var(--glow);}
.metric-val{font-size:2.2rem;font-weight:700;color:var(--teal-neon);line-height:1.1;font-family:'DM Sans',sans-serif;}
.metric-lbl{font-size:.75rem;color:var(--text-dim);letter-spacing:.1em;text-transform:uppercase;}

/* Section Headers */
.section-head{
    font-family:'DM Serif Display',serif;font-size:1.3rem;
    color:var(--teal-neon);border-left:4px solid var(--teal-neon);
    padding-left:12px;margin:32px 0 16px;background:linear-gradient(90deg,rgba(20,255,236,0.1) 0%,transparent 100%);
    padding:8px 12px 8px 16px;
}

/* Buttons */
.stButton>button{
    background:linear-gradient(135deg,var(--teal) 0%,var(--teal-dark) 100%);
    color:var(--teal-light);border:none;border-radius:8px;font-weight:600;
    box-shadow:0 4px 15px rgba(13,148,136,0.4);
}
.stButton>button:hover{background:linear-gradient(135deg,var(--teal-neon) 0%,var(--teal) 100%);color:var(--bg-dark)!important;}

/* Select boxes */
.stSelect>div>div{background:var(--bg-card)!important;border:1px solid var(--border)!important;color:var(--text)!important;}
.stMultiSelect>div>div{background:var(--bg-card)!important;border:1px solid var(--border)!important;color:var(--text)!important;}

/* DataFrames */
[data-testid="stDataFrame"]{background:var(--bg-card)!important;border:1px solid var(--border)!important;border-radius:12px;}

/* Scrollbar */
::-webkit-scrollbar{width:8px;height:8px;}
::-webkit-scrollbar-track{background:var(--bg-dark);}
::-webkit-scrollbar-thumb{background:var(--teal-dark);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:var(--teal);}

/* Alert/Info Boxes */
.alert-box{
    background:linear-gradient(135deg,rgba(239,68,68,0.1) 0%,rgba(239,68,68,0.05) 100%);
    border:1px solid var(--red);border-radius:12px;padding:16px;margin:8px 0;
    animation:pulse 2s infinite;
}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.4);}50%{box-shadow:0 0 0 10px rgba(239,68,68,0);}}

.info-box{
    background:linear-gradient(135deg,rgba(20,255,236,0.1) 0%,rgba(20,255,236,0.05) 100%);
    border:1px solid var(--teal);border-radius:12px;padding:16px;margin:8px 0;
}

/* Cards */
.radar-card{background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:20px;transition:all 0.3s ease;}
.radar-card:hover{border-color:var(--teal-neon);box-shadow:0 0 20px var(--glow);}

/* KPI with delta */
.kpi-delta-pos{color:var(--teal-neon)!important;}
.kpi-delta-neg{color:var(--red)!important;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# 📊 HELPER FUNCTIONS
# ============================================================================
def kpi(col, label, val, delta=None, fmt=None, key=None):
    """Custom KPI card matching reference_app.py style"""
    # Handle non-numeric values (strings, None, etc.)
    if val is None:
        display = "N/A"
    elif isinstance(val, str):
        display = str(val)  # Use string as-is
    elif fmt == "money":
        display = f"${val:,.0f}" if val < 10000 else f"${val / 1e3:,.1f}K"
    elif fmt == "pct":
        display = f"{val:.1f}%"
    elif fmt == "dec":
        display = f"{val:.3f}"
    else:
        display = f"{val:,}"

    delta_html = ""
    if delta is not None:
        delta_color = "var(--teal-neon)" if delta > 0 else "var(--red)"
        delta_icon = "▲" if delta > 0 else "▼"
        delta_html = f"<div style='color:{delta_color};font-size:.85rem;margin-top:4px;'>{delta_icon} {abs(delta):.1f}%</div>"

    col.markdown(
        f"<div class='metric-card'><div class='metric-lbl'>{label}</div><div class='metric-val'>{display}</div>{delta_html}</div>",
        unsafe_allow_html=True,
    )


def plot_radar(categories, values, title, color="#14ffec"):
    """Radar chart matching reference_app.py style"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            line_color=color,
            fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.3)",
            name=title,
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 1], tickfont=dict(color="#94a3b8"), gridcolor="#1e3a35"
            ),
            bgcolor="#0a0f0d",
        ),
        paper_bgcolor="#0a0f0d",
        margin=dict(t=30, b=20, l=20, r=20),
        showlegend=False,
    )
    return fig


# ============================================================================
# 🧠 STATE MANAGEMENT
# ============================================================================
if "df" not in st.session_state:
    st.session_state.df = None
if "context" not in st.session_state:
    st.session_state.context = {}
if "pipeline_completed" not in st.session_state:
    st.session_state.pipeline_completed = {i: False for i in range(1, 11)}


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


# ============================================================================
# 📥 AUTO-LOAD SAMPLE DATA
# ============================================================================
@st.cache_data
def load_sample_data():
    """Auto-load synthetic sample data if exists"""
    sample_path = Path("data/sales_data.csv")
    if sample_path.exists():
        return pd.read_csv(sample_path)
    return None


sample_df = load_sample_data()
if sample_df is not None and st.session_state.df is None:
    st.session_state.df = sample_df
    st.session_state.sample_loaded = True
else:
    st.session_state.sample_loaded = False


# ============================================================================
# 🎛️ SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("## 📊 Data Science")
    st.markdown("### Platform")
    st.markdown("---")

    st.markdown("**Navigation**")
    page = st.radio(
        "Go to",
        [
            "Data Hub",
            "EDA Dashboard",
            "Feature Lab",
            "Hypothesis Lab",
            "ML Studio",
            "Reports",
            "API Explorer",
            "Executive Summary",
        ],
        label_visibility="collapsed",
        index=0,
    )
    st.markdown("---")

    # Data source selector
    st.markdown("**Data Source**")
    data_source = st.radio(
        "Select",
        ["Sample Data", "Upload New"],
        horizontal=True,
        index=0 if st.session_state.sample_loaded else 1,
    )

    if data_source == "Upload New":
        uploaded_file = st.file_uploader(
            "Upload CSV/Excel",
            type=["csv", "xlsx"],
            help="Upload your dataset",
        )
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success(
                    f"✅ Loaded: {st.session_state.df.shape[0]} × {st.session_state.df.shape[1]}"
                )
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # Quick stats
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"**Current Data**")
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.metric("Missing", f"{df.isnull().sum().sum():,}")


# ============================================================================
# 📄 PAGE ROUTING
# ============================================================================

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: DATA HUB
# ═══════════════════════════════════════════════════════════════════���═���═════════
if page == "Data Hub":
    st.markdown("# 📊 Data Hub")
    st.markdown(
        "<span style='color:var(--text-muted)'>Data ingestion & quality overview</span>",
        unsafe_allow_html=True,
    )

    if st.session_state.df is not None:
        df = st.session_state.df

        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "Total Rows", df.shape[0])
        kpi(k2, "Columns", df.shape[1])

        missing_total = df.isnull().sum().sum()
        missing_pct = (missing_total / (df.shape[0] * df.shape[1])) * 100
        kpi(k3, "Missing Values", missing_pct, fmt="pct")

        dupes = df.duplicated().sum()
        kpi(k4, "Duplicates", dupes)

        # Data Quality Section - LARGER COMPONENTS
        st.markdown("<div class='section-head'>Data Quality Overview</div>", unsafe_allow_html=True)

        # Row 1: Missing Gauge + Duplicates (side by side)
        q1, q2 = st.columns(2)

        # Missing gauge - LARGER
        with q1:
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=missing_pct,
                    title={"text": "Missing Data %", "font": {"size": 24}},
                    number={"font": {"size": 48}},
                    gauge={
                        "axis": {"range": [0, 100], "tickfont": {"size": 16}},
                        "bar": {"color": "#FF4B4B" if missing_pct > 10 else "#14ffec"},
                        "steps": [
                            {"range": [0, 5], "color": "#1E1E1E"},
                            {"range": [5, 10], "color": "#262730"},
                            {"range": [10, 100], "color": "#3E1E1E"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", margin={"t": 50})
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Duplicates indicator - LARGER
        with q2:
            fig_dup = go.Figure(
                go.Indicator(
                    mode="number",
                    value=dupes,
                    title={"text": "Duplicate Rows", "font": {"size": 24}},
                    number={"font": {"size": 48, "color": "#ef4444" if dupes > 0 else "#14ffec"}},
                )
            )
            fig_dup.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", margin={"t": 50})
            st.plotly_chart(fig_dup, use_container_width=True)

        # Row 2: Data types pie - FULL WIDTH LARGER
        st.markdown("### 📊 Data Types Distribution")
        dtype_counts = df.dtypes.astype(str).value_counts()
        fig_pie = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index,
            title="Column Types",
            color_discrete_sequence=["#14ffec", "#ef4444", "#0ea5e9", "#f59e0b"],
        )
        fig_pie.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=16),
            title_font_size=24,
            title_x=0.5,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Column Details
        st.markdown("<div class='section-head'>Column Details</div>", unsafe_allow_html=True)

        col_detail = st.selectbox("Select Column", df.columns)
        col_data = df[col_detail]

        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Type", str(col_data.dtype))  # Passes string directly now
        kpi(c2, "Unique", col_data.nunique())
        kpi(c3, "Missing", col_data.isnull().sum())
        kpi(c4, "Missing %", (col_data.isnull().sum() / len(df)) * 100, fmt="pct")

        # Data preview
        st.markdown("<div class='section-head'>Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, "data_hub.csv", "text/csv")

    else:
        st.markdown(
            "<div class='info-box'>📤 Upload a dataset or use Sample Data from sidebar</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EDA DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "EDA Dashboard":
    st.markdown("# 📈 EDA Dashboard")
    st.markdown(
        "<span style='color:var(--text-muted)'>Deep exploratory data analysis</span>",
        unsafe_allow_html=True,
    )

    if st.session_state.df is not None:
        df = st.session_state.df
        eda = agents["eda"]

        # Settings
        c1, c2 = st.columns(2)
        with c1:
            target = st.selectbox(
                "Target Variable",
                df.columns,
                index=len(df.columns) - 1 if len(df.columns) > 0 else 0,
            )
        with c2:
            corr_method = st.radio("Correlation Method", ["pearson", "spearman"], horizontal=True)

        # Analyze
        results = eda.analyze(df, target, corr_method)

        # KPIs
        st.markdown("<div class='section-head'>Key Statistics</div>", unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        kpi(
            s1,
            "Target Mean",
            df[target].mean() if pd.api.types.is_numeric_dtype(df[target]) else "N/A",
            fmt="dec",
        )
        kpi(
            s2,
            "Target Std",
            df[target].std() if pd.api.types.is_numeric_dtype(df[target]) else "N/A",
            fmt="dec",
        )
        kpi(s3, "Skewness", results["target_info"]["skew"], fmt="dec")
        kpi(s4, "Unique Values", results["target_info"]["unique_values"])

        # Correlation Heatmap
        st.markdown("<div class='section-head'>Correlation Analysis</div>", unsafe_allow_html=True)

        corr_matrix = df.select_dtypes(include="number").corr(method=corr_method)
        fig_heat = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            title=f"{corr_method.title()} Correlation",
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Top Correlations
        st.markdown("<div class='section-head'>Top Correlations</div>", unsafe_allow_html=True)
        corr_pairs = eda.get_correlation_pairs(df, corr_method, 0.3)
        # Drop NaN and filter valid pairs (column name is lowercase)
        corr_pairs = corr_pairs.dropna(subset=["correlation"])

        if not corr_pairs.empty and len(corr_pairs) > 0:
            top_corr = corr_pairs.head(10).reset_index(drop=True)
            # Use lowercase column names
            fig_corr = px.bar(
                top_corr,
                x="correlation",
                y="variable_1",
                orientation="h",
                color="correlation",
                color_continuous_scale=["#ef4444", "#14ffec"],
                title="Top 10 Correlation Pairs",
            )
            fig_corr.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No significant correlations found (threshold: 0.3)")

        # Distribution
        st.markdown("<div class='section-head'>Distributions</div>", unsafe_allow_html=True)

        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            dist_col = st.selectbox("Select Column", num_cols, index=0)

            # Violin plot
            if df.select_dtypes(include="object").columns.any():
                cat_cols = df.select_dtypes(include="object").columns.tolist()
                cat_col = st.selectbox("Group by", cat_cols)
                fig_violin = px.violin(
                    df,
                    y=dist_col,
                    x=cat_col,
                    box=True,
                    points="outliers",
                    title=f"Distribution: {dist_col}",
                )
                fig_violin.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_violin, use_container_width=True)
            else:
                # Histogram
                fig_hist = px.histogram(df, x=dist_col, nbins=30, title=f"Distribution: {dist_col}")
                fig_hist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("⚠️ Load data first from Data Hub")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 3: FEATURE LAB
# ═══════════════════════════════════════════════════════════════════════
elif page == "Feature Lab":
    st.markdown("# ⚙️ Feature Lab")
    st.markdown(
        "<span style='color:var(--text-muted)'>Feature engineering & transformations</span>",
        unsafe_allow_html=True,
    )

    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        fe = agents["features"]

        # Feature creation options
        with st.expander("📅 Date Features", expanded=False):
            date_cols = df.select_dtypes(include="datetime").columns.tolist()
            if date_cols:
                selected_dates = st.multiselect("Date Columns", date_cols)
                for col in selected_dates:
                    df[col] = pd.to_datetime(df[col])
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                st.success(f"✅ Created date features for {len(selected_dates)} columns")
            else:
                st.info("No datetime columns found")

        with st.expander("📐 Ratio Features"):
            num_cols = df.select_dtypes(include="number").columns.tolist()
            c1, c2 = st.columns(2)
            with c1:
                num_col = st.selectbox("Numerator", num_cols, key="ratio_num")
            with c2:
                den_col = st.selectbox(
                    "Denominator", [c for c in num_cols if c != num_col], key="ratio_den"
                )

            if st.button("Create Ratio"):
                df[f"{num_col}_{den_col}_ratio"] = df[num_col] / (df[den_col] + 1e-9)
                st.success(f"✅ Created {num_col}_{den_col}_ratio")

        with st.expander("🔢 Interaction Features"):
            num_cols = df.select_dtypes(include="number").columns.tolist()
            selected = st.multiselect("Select Columns", num_cols, default=num_cols[:3])

            if st.button("Create Interactions") and selected:
                orig_cols = list(df.columns)
                df = fe.create_interaction_features(
                    df, selected, include_products=True, include_powers=True
                )
                new_cols = [c for c in df.columns if c not in orig_cols]
                st.success(f"✅ Created {len(new_cols)} interaction features")

        with st.expander("📊 Scaling"):
            scale_method = st.selectbox(
                "Method", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"]
            )
            if scale_method != "None" and st.button("Apply Scaling"):
                df = fe.transform(df, {"scaling": scale_method})
                st.success(f"✅ Applied {scale_method}")

        with st.expander("🏷️ Encoding"):
            cat_cols = df.select_dtypes(include="object").columns.tolist()
            if cat_cols:
                encode_method = st.selectbox("Method", ["Label", "OneHot"])
                target_col = st.selectbox(
                    "Target (optional)", ["None"] + df.columns.tolist(), index=0
                )
                if st.button("Apply Encoding"):
                    target = target_col if target_col != "None" else None
                    df = fe.apply_encoding(df, encode_method, target)
                    st.success(f"✅ Applied {encode_method} encoding")
            else:
                st.info("No categorical columns found")

        # Show result
        st.markdown("<div class='section-head'>Engineered Data</div>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        # Save to session
        if st.button("💾 Save to Pipeline"):
            st.session_state.df = df
            st.session_state.pipeline_completed[3] = True
            st.success("✅ Data saved!")

    else:
        st.warning("⚠️ Load data first")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 4: HYPOTHESIS LAB
# ═══════════════════════════════════════════════════════════════════════
elif page == "Hypothesis Lab":
    st.markdown("# 🧪 Hypothesis Lab")
    st.markdown(
        "<span style='color:var(--text-muted)'>Statistical testing & inference</span>",
        unsafe_allow_html=True,
    )

    if st.session_state.df is not None:
        df = st.session_state.df

        # Hypothesis inputs
        h0 = st.text_input("H₀ (Null Hypothesis)", "There is no significant difference...")
        h1 = st.text_input("H₁ (Alternative)", "There is a significant difference...")

        # Test settings
        c1, c2 = st.columns(2)
        with c1:
            test_type = st.selectbox("Test Type", ["t-test", "chi-square", "anova", "mann-whitney"])
        with c2:
            alpha = st.slider("Significance (α)", 0.01, 0.10, 0.05)

        # Column selection
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        col_a = st.selectbox("Column A / Variable", num_cols + cat_cols)

        col_b = None
        if test_type in ["chi-square", "anova", "mann-whitney"]:
            col_b = st.selectbox("Column B (Grouping)", ["None"] + num_cols + cat_cols)

        # Run test
        if st.button("🔬 Run Test"):
            config = {"column_a": col_a, "alpha": alpha}
            if col_b and col_b != "None":
                config["column_b"] = col_b

            result = agents["stats"].run_test(df, test_type, config)

            if "statistic" in result:
                st.session_state.context["stats_result"] = result
                st.session_state.pipeline_completed[4] = True

                # Results display
                st.markdown("<div class='section-head'>Test Results</div>", unsafe_allow_html=True)

                r1, r2 = st.columns(2)
                kpi(r1, "Test Statistic", result["statistic"], fmt="dec")
                kpi(r2, "P-Value", result["p_value"], fmt="dec")

                # Interpretation
                p_val = result["p_value"]
                decision = "Reject H₀" if p_val < alpha else "Fail to Reject H₀"

                if p_val < alpha:
                    st.markdown(
                        f"<div class='info-box'><strong>✅ {decision}</strong><br>p-value ({p_val:.4f}) < α ({alpha})<br>The result is statistically significant.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='alert-box'><strong>❌ {decision}</strong><br>p-value ({p_val:.4f}) ≥ α ({alpha})<br>No significant difference found.</div>",
                        unsafe_allow_html=True,
                    )

                # JSON result
                st.json(result)
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")

    else:
        st.warning("⚠️ Load data first")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 5: ML STUDIO
# ═══════════════════════════════════════════════════════════════════════
elif page == "ML Studio":
    st.markdown("# 🤖 ML Studio")
    st.markdown(
        "<span style='color:var('text-muted')>Machine learning training & evaluation</span>",
        unsafe_allow_html=True,
    )

    if st.session_state.df is not None:
        df = st.session_state.df

        # Settings
        c1, c2, c3 = st.columns(3)
        with c1:
            target = st.selectbox("Target", df.columns)
        with c2:
            problem_type = st.radio("Problem", ["Classification", "Regression"], horizontal=True)
        with c3:
            model_choice = st.selectbox(
                "Algorithm",
                ["Random Forest", "Gradient Boosting", "Logistic Regression"]
                if problem_type == "Classification"
                else ["Random Forest", "Gradient Boosting", "Ridge", "Lasso"],
            )

        c4, c5 = st.columns(2)
        with c4:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        with c5:
            seed = st.number_input("Random Seed", value=42)

        # Train
        if st.button("🚀 Train Model"):
            # Clean data: drop rows with missing target or non-numeric columns
            df_clean = df.dropna(subset=[target]).copy()

            # Keep only numeric columns for features
            numeric_cols = df_clean.select_dtypes(include="number").columns.tolist()
            if target not in numeric_cols:
                st.error(f"⚠️ Target '{target}' must be numeric")
            else:
                df_clean = df_clean[numeric_cols]

                # Also drop any remaining NaN in features
                df_clean = df_clean.dropna()

                if len(df_clean) < 10:
                    st.error("⚠️ Not enough data after cleaning. Need at least 10 rows.")
                else:
                    config = {
                        "target": target,
                        "test_size": test_size,
                        "random_seed": seed,
                        "model_type": model_choice,
                        "problem_type": problem_type,
                    }

                    with st.spinner("Training..."):
                        result = agents["model"].train_and_evaluate(df_clean, target, config)

            if result:
                st.session_state.context["model_metrics"] = result["metrics"]
                st.session_state.pipeline_completed[5] = True

                # Metrics
                st.markdown(
                    "<div class='section-head'>Model Performance</div>", unsafe_allow_html=True
                )

                m = result["metrics"]
                cols = st.columns(len(m) - 1 if "classification_report" in m else len(m))

                for i, (k, v) in enumerate(m.items()):
                    if k != "classification_report":
                        with cols[i % len(cols)]:
                            kpi(
                                cols[i % len(cols)],
                                k.upper(),
                                v if not isinstance(v, (list, dict)) else str(v)[:10],
                                fmt="dec",
                            )

                # Confusion Matrix (Classification)
                if problem_type == "Classification" and "confusion_matrix" in m:
                    st.markdown(
                        "<div class='section-head'>Confusion Matrix</div>", unsafe_allow_html=True
                    )

                    cm = m["confusion_matrix"]
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale=["#0d1411", "#14ffec"],
                        title="Confusion Matrix",
                    )
                    fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                    st.plotly_chart(fig_cm, use_container_width=True)

                # ROC Curve
                if "roc_auc" in m:
                    st.markdown("<div class='section-head'>ROC Curve</div>", unsafe_allow_html=True)
                    st.metric("ROC-AUC", f"{m['roc_auc']:.4f}")

                    if "roc_curve" in m:
                        roc = m["roc_curve"]
                        fig_roc = go.Figure()
                        fig_roc.add_trace(
                            go.Scatter(
                                x=roc["fpr"],
                                y=roc["tpr"],
                                mode="lines",
                                name=f"ROC (AUC={m['roc_auc']:.3f})",
                                line=dict(color="#14ffec", width=2),
                            )
                        )
                        fig_roc.add_trace(
                            go.Scatter(
                                x=[0, 1],
                                y=[0, 1],
                                mode="lines",
                                name="Random",
                                line=dict(color="gray", dash="dash"),
                            )
                        )
                        fig_roc.update_layout(
                            xaxis_title="False Positive Rate",
                            yaxis_title="True Positive Rate",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white"),
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)

                # Feature Importance
                if result.get("feature_importances"):
                    st.markdown(
                        "<div class='section-head'>Feature Importance</div>", unsafe_allow_html=True
                    )

                    fi = result["feature_importances"]
                    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
                    fi_df = fi_df.sort_values("Importance", ascending=True).tail(10)

                    fig_fi = px.bar(
                        fi_df,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title="Top 10 Features",
                    )
                    fig_fi.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

                st.success("✅ Model trained!")
    else:
        st.warning("⚠️ Load data first")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 6: REPORTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "Reports":
    st.markdown("# 📝 Reports")
    st.markdown(
        "<span style='color:var(--text-muted)'>Generate detailed insight reports</span>",
        unsafe_allow_html=True,
    )

    # Report settings
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Report Title", "Data Science Analysis Report")
    with col2:
        audience = st.selectbox("Audience", ["Technical", "Executive", "General"])

    # Dataset info if available
    dataset_info = {}
    if st.session_state.df is not None:
        df = st.session_state.df
        dataset_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
        }

    # Build context from session state
    context = {}

    # Data quality
    if dataset_info:
        missing_pct = (
            dataset_info["missing"] / (dataset_info["rows"] * dataset_info["columns"])
        ) * 100
        context["data_quality"] = {
            "missing_pct": missing_pct,
            "duplicates": dataset_info["duplicates"],
            "quality_score": 100 - missing_pct,
        }

    # EDA results from session state
    if st.session_state.context.get("eda_results"):
        eda = st.session_state.context["eda_results"]
        context["eda"] = eda
        # Add top correlations
        if st.session_state.df is not None and "eda_results" in st.session_state.context:
            try:
                from ds_platform.agents.eda import EDAAgent

                eda_agent = EDAAgent()
                corr = eda_agent.get_correlation_pairs(st.session_state.df, "pearson", 0.3).head(5)
                context["eda"]["top_correlations"] = [
                    {"var1": r["variable_1"], "var2": "variable_2", "corr": r["correlation"]}
                    for _, r in corr.iterrows()
                ]
            except:
                pass

    # Model metrics
    if st.session_state.context.get("model_metrics"):
        context["model"] = st.session_state.context["model_metrics"]

    # Stats results
    if st.session_state.context.get("stats_result"):
        context["stats"] = st.session_state.context["stats_result"]

    # Generate report
    if st.button("📜 Generate Detailed Report", type="primary"):
        path = agents["report"].generate_report(
            title=title,
            audience=audience,
            context=context,
            dataset_info=dataset_info if dataset_info else None,
        )

        if path:
            st.session_state.pipeline_completed[6] = True
            st.success(f"✅ Report saved to {path}")

            # Show report preview
            with open(path, "r", encoding="utf-8") as f:
                report_content = f.read()

            st.markdown("<div class='section-head'>Report Preview</div>", unsafe_allow_html=True)
            st.markdown(report_content, unsafe_allow_html=True)

    # Download options
    st.markdown("<div class='section-head'>Download Options</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        # Generate and offer markdown download
        if st.session_state.df is not None:
            # Get the actual report content
            path = agents["report"].generate_report(
                title=title,
                audience=audience,
                context=context,
                dataset_info=dataset_info if dataset_info else None,
            )
            if path and Path(path).exists():
                with open(path, "rb") as f:
                    st.download_button(
                        "📥 Download Markdown Report",
                        data=f.read(),
                        file_name="ds_analysis_report.md",
                        mime="text/markdown",
                    )
        else:
            st.info("👆 Generate a report first")

    with col2:
        # Try to generate PDF
        try:
            from ds_platform.agents.report import convert_markdown_to_pdf

            # First generate markdown
            if st.session_state.df is not None:
                md_path = agents["report"].generate_report(
                    title=title,
                    audience=audience,
                    context=context,
                    dataset_info=dataset_info if dataset_info else None,
                )

                # Try convert to PDF
                pdf_path = convert_markdown_to_pdf(md_path)

                if pdf_path and Path(pdf_path).exists():
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "📥 Download PDF Report",
                            data=f.read(),
                            file_name="ds_analysis_report.pdf",
                            mime="application/pdf",
                        )
                else:
                    st.warning("PDF requires weasyprint: pip install weasyprint")
        except Exception as e:
            st.warning(f"PDF not available: {str(e)[:50]}")

    # Show what data will be included
    if context:
        st.markdown("<div class='section-head'>Report Contents</div>", unsafe_allow_html=True)

        # Preview sections
        sections = []
        if dataset_info:
            sections.append("✅ Dataset Overview")
        if context.get("eda"):
            sections.append("✅ EDA Analysis")
        if context.get("model"):
            sections.append("✅ ML Model Performance")
        if context.get("stats"):
            sections.append("✅ Statistical Tests")

        if sections:
            for s in sections:
                st.markdown(f"- {s}")
        else:
            st.info("Run pipeline stages first to include their results")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 7: API EXPLORER
# ═══════════════════════════════════════════════════════════════════════
elif page == "API Explorer":
    st.markdown("# 🌐 API Explorer")
    st.markdown(
        "<span style='color:var(--text-muted)'>Fetch data from external APIs</span>",
        unsafe_allow_html=True,
    )

    # API settings
    url = st.text_input("API Endpoint", placeholder="https://api.example.com/data")

    c1, c2 = st.columns(2)
    with c1:
        method = st.selectbox("Method", ["GET", "POST"])
    with c2:
        api_key = st.text_input("API Key (optional)", type="password")

    # Headers
    with st.expander("Headers"):
        header_key = st.text_input("Header Key")
        header_val = st.text_input("Header Value")

    # Fetch
    if st.button("📡 Fetch Data"):
        if url:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            df_api = agents["api"].fetch(url, headers)

            if not df_api.empty:
                st.success(f"✅ Fetched {len(df_api)} records")

                # Preview
                st.dataframe(df_api.head(), use_container_width=True)

                # Load to pipeline
                if st.button("📥 Load to Pipeline"):
                    st.session_state.df = df_api
                    st.session_state.pipeline_completed[7] = True
                    st.success("✅ Data loaded! Go to Data Hub.")
        else:
            st.error("⚠️ Enter a URL")

    # Preset APIs
    st.markdown("<div class='section-head'>Quick APIs</div>", unsafe_allow_html=True)

    if st.button("📊 Load Crypto Sample"):
        # Demo: just use sample data
        sample = load_sample_data()
        if sample is not None:
            st.session_state.df = sample
            st.success("✅ Sample data loaded")


# ═══════════════════════════════════════════════════════════════════════
# PAGE 8: EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════
elif page == "Executive Summary":
    st.markdown("# 📋 Executive Summary")
    st.markdown(
        "<span style='color:var(--text-muted)'>CEO-ready one-pager snapshot</span>",
        unsafe_allow_html=True,
    )

    if st.session_state.df is not None:
        df = st.session_state.df

        # Header KPIs
        total_rows = len(df)
        total_cols = df.shape[1]
        missing = df.isnull().sum().sum()
        missing_pct = (missing / (total_rows * total_cols)) * 100

        numeric_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include="object").columns

        d1, d2, d3, d4 = st.columns(4)
        kpi(d1, "Total Rows", total_rows)
        kpi(d2, "Features", total_cols)
        kpi(d3, "Numeric", len(numeric_cols))
        kpi(d4, "Missing %", missing_pct, fmt="pct")

        # Summary Cards
        st.markdown(
            """
            <style>
            .exec-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px;}
            .exec-card{
                background:linear-gradient(135deg,var(--bg-card) 0%,#0f1f1c 100%);
                border:1px solid var(--border);border-radius:12px;padding:20px;text-align:center;
            }
            .exec-val{font-size:2rem;font-weight:700;color:var(--teal-neon);}
            .exec-lbl{font-size:0.8rem;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.1em;}
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Key Metrics Grid
        st.markdown("<div class='exec-grid'>", unsafe_allow_html=True)
        cols = st.columns(4)

        with cols[0]:
            st.markdown(
                "<div class='exec-card'><div class='exec-lbl'>Numeric Features</div><div class='exec-val'>{num}</div></div>".format(
                    num=len(numeric_cols)
                ),
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                "<div class='exec-card'><div class='exec-lbl'>Categorical</div><div class='exec-val'>{cat}</div></div>".format(
                    cat=len(cat_cols)
                ),
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                "<div class='exec-card'><div class='exec-lbl'>Missing Values</div><div class='exec-val'>{mis:,}</div></div>".format(
                    mis=missing
                ),
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.markdown(
                "<div class='exec-card'><div class='exec-lbl'>Data Quality</div><div class='exec-val'>{dq:.1f}%</div></div>".format(
                    dq=100 - missing_pct
                ),
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Visualizations
        c1, c2 = st.columns(2)

        with c1:
            if len(numeric_cols) > 0:
                # Top correlations
                corr = df[numeric_cols].corr().abs()
                corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                corr = corr.stack().sort_values(ascending=False).head(5)

                if len(corr) > 0:
                    corr_df = pd.DataFrame(
                        {"Pair": [f"{a}x{b}" for a, b in corr.index], "Correlation": corr.values}
                    )
                    fig = px.bar(
                        corr_df,
                        x="Correlation",
                        y="Pair",
                        orientation="h",
                        title="Top Correlations",
                        color="Correlation",
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with c2:
            if len(cat_cols) > 0:
                # Category distribution
                cat_col = cat_cols[0]
                cat_counts = df[cat_col].value_counts().head(5)
                fig = px.pie(
                    values=cat_counts.values,
                    names=cat_counts.index,
                    title=f"Top Categories: {cat_col}",
                )
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig, use_container_width=True)

        # Model Performance Summary
        if st.session_state.context.get("model_metrics"):
            st.markdown("<div class='section-head'>ML Performance</div>", unsafe_allow_html=True)

            m = st.session_state.context["model_metrics"]
            m_cols = st.columns(len(m))
            for i, (k, v) in enumerate(m.items()):
                if isinstance(v, (int, float)) and k != "confusion_matrix":
                    with m_cols[i % len(m_cols)]:
                        kpi(m_cols[i % len(m_cols)], k.upper(), v, fmt="dec")

        # Print button
        st.markdown("---")
        st.markdown(
            """
        <button onclick="window.print()" style="background:var(--teal);color:white;padding:10px 20px;border:none;border-radius:5px;cursor:pointer;">🖨️ Print Report</button>
        """,
            unsafe_allow_html=True,
        )

    else:
        st.warning("⚠️ No data loaded. Please upload or select sample data.")

        # Demo button
        if st.button("📊 Load Demo Data"):
            sample = load_sample_data()
            if sample is not None:
                st.session_state.df = sample
                st.rerun()
            else:
                st.error("No sample data found. Upload a dataset first.")
