import streamlit as st
from core.arrow_fix import make_arrow_compatible


if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
from core.data_cleaner import clean_data

st.title("🧹 Data Cleaning")

if "raw_df" not in st.session_state:
    st.warning("Please upload dataset first.")
    st.stop()

df = st.session_state.raw_df.copy()

cleaned_df, cleaning_log = clean_data(df)
for col in cleaned_df.select_dtypes(include=["object"]).columns:
    cleaned_df[col] = cleaned_df[col].astype(str)

st.session_state.cleaned_df = cleaned_df

st.success("✅ Automatic Data Cleaning Completed")

st.subheader("📝 Cleaning Summary")

for log in cleaning_log:
    st.write("•", log)

st.divider()

st.subheader("Preview of Cleaned Dataset (First 5 Rows)")
st.dataframe(cleaned_df.head())

