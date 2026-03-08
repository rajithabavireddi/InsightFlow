import streamlit as st

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
from core.dataset_analyzer import analyze_dataset
from core.quality_score import calculate_quality_score, interpret_score

st.set_page_config(layout="wide")

st.title("📊 Data Intelligence Dashboard")
st.markdown("Comprehensive Dataset Analysis & Quality Assessment")

# =====================================================
# CHECK SESSION
# =====================================================

if "raw_df" not in st.session_state:
    st.warning("Please upload a dataset first.")
    st.stop()

df = st.session_state.raw_df

# =====================================================
# DATASET ANALYSIS
# =====================================================

dataset_info = analyze_dataset(df)
quality_score = calculate_quality_score(df)
quality_interpretation = interpret_score(quality_score)

# Store in session
st.session_state.dataset_info = dataset_info
st.session_state.quality_score = quality_score

# =====================================================
# DATASET OVERVIEW KPIs
# =====================================================

st.markdown("## 📌 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Total Rows",
    value=f"{dataset_info['Rows']:,}"
)

col2.metric(
    label="Total Columns",
    value=dataset_info["Columns"]
)

col3.metric(
    label="Missing Values",
    value=f"{dataset_info['Missing Values']:,}"
)

col4.metric(
    label="Duplicate Rows",
    value=f"{dataset_info['Duplicate Rows']:,}"
)

st.divider()

# =====================================================
# FEATURE COMPOSITION
# =====================================================

st.markdown("## 🧠 Feature Composition")

col5, col6 = st.columns(2)

col5.metric(
    label="Numeric Features",
    value=dataset_info["Numeric Features"]
)

col6.metric(
    label="Categorical Features",
    value=dataset_info["Categorical Features"]
)

# 🔹 NEW ADDITION — FEATURE NAMES DISPLAY
with st.expander("🔍 View Feature Names"):

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    st.write("### Numeric Columns")
    if numeric_cols:
        st.write(numeric_cols)
    else:
        st.write("None")

    st.write("### Categorical Columns")
    if categorical_cols:
        st.write(categorical_cols)
    else:
        st.write("None")

st.divider()

# =====================================================
# QUALITY SCORE SECTION
# =====================================================

st.markdown("## ⭐ Data Quality Score")

col7, col8 = st.columns([1, 2])

col7.metric(
    label="Quality Score",
    value=f"{quality_score}/100"
)

# Quality Interpretation Styling
if quality_score >= 85:
    col8.success(quality_interpretation)
elif quality_score >= 70:
    col8.warning(quality_interpretation)
else:
    col8.error(quality_interpretation)

st.divider()

# =====================================================
# RAW DATA PREVIEW
# =====================================================

with st.expander("🔍 Preview Dataset"):
    st.dataframe(df.head())
