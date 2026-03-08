import streamlit as st
import pandas as pd
from core.arrow_fix import make_arrow_compatible

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()

st.title("📂 Upload Dataset")

# =====================================================
# ROLE CHECK
# =====================================================

if "user_role" not in st.session_state:
    st.warning("Please select user mode first.")
    st.stop()

# =====================================================
# FILE UPLOAD
# =====================================================

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, low_memory=False)
    df = df.convert_dtypes()
    df = make_arrow_compatible(df)
    # Store datasets in session
    st.session_state.raw_df = df
    st.session_state["original_df"] = df.copy()   # for business insights
    st.session_state.cleaned_df = df.copy()

    st.success("✅ Dataset Uploaded Successfully!")

    st.dataframe(df.head(), width="stretch")

else:
    st.info("Please upload a CSV file to continue.")