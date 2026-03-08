import streamlit as st

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()
import numpy as np

st.title("📊 Exploratory Data Analysis")

required = ["cleaned_df", "user_role"]

for key in required:
    if key not in st.session_state:
        st.warning("Please complete previous steps first.")
        st.stop()

df = st.session_state.cleaned_df
role = st.session_state.user_role

st.subheader("Dataset Shape")
st.write(df.shape)

# Business & Non Technical
if role in ["Non-Technical User", "Business User"]:
    st.subheader("📌 Basic Statistical Summary")
    st.write(df.describe())

# Data Analyst
elif role == "Data Analyst":
    st.subheader("📊 Statistical Summary")
    st.write(df.describe())

    st.subheader("Correlation Matrix")
    st.write(df.corr(numeric_only=True))

# Data Scientist
elif role == "Data Scientist":
    st.subheader("📊 Detailed Statistical Summary")
    st.write(df.describe(include="all"))

    st.subheader("Correlation Matrix")
    st.write(df.corr(numeric_only=True))

    st.subheader("Feature Variance")
    st.write(df.var(numeric_only=True))
