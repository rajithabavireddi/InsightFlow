import streamlit as st
import pandas as pd
import numpy as np

# =====================================================
# LOGIN CHECK
# =====================================================

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()

st.title("🧠 AI Insights Dashboard")

# =====================================================
# USER ROLE CHECK
# =====================================================

user_role = st.session_state.get("user_role", "Data Scientist")

# =====================================================
# CHECK REQUIRED DATA
# =====================================================

if "results_df" not in st.session_state:
    st.error("❌ Model results not found in session.")
    st.stop()

results_df = st.session_state.get("results_df")
best_model_name = st.session_state.get("best_model_name")
problem_type = st.session_state.get("problem_type")
target_column = st.session_state.get("target_column")

if results_df is None:
    st.stop()

# =====================================================
# FIX ARROW SERIALIZATION FOR RESULTS_DF
# =====================================================

results_df = results_df.copy()

for col in results_df.select_dtypes(include=["object"]).columns:
    results_df[col] = results_df[col].astype(str)

st.success("✅ Model Insights Loaded Successfully")

# =====================================================
# GET BEST MODEL ROW
# =====================================================

best_row = results_df[results_df["Model"] == best_model_name].iloc[0]

# SAFE SCORE EXTRACTION

train_score = best_row["Train Score"] if "Train Score" in results_df.columns else None
val_score = best_row["Validation Score"] if "Validation Score" in results_df.columns else None
test_score = best_row["Test Score"] if "Test Score" in results_df.columns else None

if val_score is None and "Accuracy" in results_df.columns:
    val_score = best_row["Accuracy"]

if val_score is None and "R2 Score" in results_df.columns:
    val_score = best_row["R2 Score"]

training_time = None

if "Training Time (sec)" in results_df.columns:
    training_time = best_row["Training Time (sec)"]

gap = 0
if train_score is not None and val_score is not None:
    gap = train_score - val_score

# =====================================================
# 1️⃣ DATA SCIENTIST / DATA ANALYST VIEW
# =====================================================

if user_role in ["Data Scientist", "Data Analyst"]:

    st.header("📊 Model Leaderboard")

    st.dataframe(results_df, width='stretch')

    st.subheader("🏆 Best Model")

    st.write(f"**Model:** {best_model_name}")

    if train_score:
        st.write(f"Train Score: {train_score}")

    if val_score:
        st.write(f"Validation Score: {val_score}")

    if test_score:
        st.write(f"Test Score: {test_score}")

    # --------------------------------------------------
    # Executive AI Summary
    # --------------------------------------------------

    st.subheader("📊 Executive AI Summary")

    st.write(f"""
    • Best performing model: **{best_model_name}**  
    • Validation Score: **{round(val_score,3)}**  
    • Model reliability gap: **{round(gap,3)}**
    """)

    if gap > 0.1:
        st.warning("⚠ Potential overfitting detected.")
    else:
        st.success("✅ Model generalization looks stable.")

    # --------------------------------------------------
    # Prediction Confidence
    # --------------------------------------------------

    st.subheader("📉 Prediction Confidence")

    confidence = round(val_score * 100,2)

    st.metric("Prediction Reliability (%)", confidence)

    if confidence > 90:
        st.success("High confidence predictions.")
    elif confidence > 75:
        st.info("Moderate confidence predictions.")
    else:
        st.warning("Prediction confidence is limited.")

    # --------------------------------------------------
    # Model Explainability
    # --------------------------------------------------

    st.subheader("🧠 Model Explainability")

    if "feature_importance" in st.session_state:

        importance = st.session_state["feature_importance"]

        if len(importance) > 0:

            importance_df = pd.DataFrame({
                "Feature": list(range(len(importance))),
                "Importance": importance
            }).sort_values("Importance", ascending=False)

            st.bar_chart(importance_df.set_index("Feature"))

    st.info("Feature importance shows which variables most influence predictions.")

    # --------------------------------------------------
    # AI Recommendations
    # --------------------------------------------------

    st.subheader("🤖 AI Recommendations")

    if gap > 0.1:
        st.write("• Apply stronger regularization.")
        st.write("• Try cross validation tuning.")
        st.write("• Consider removing noisy features.")

    else:
        st.write("• Model is stable for deployment.")
        st.write("• Monitor drift over time.")

# =====================================================
# 2️⃣ NON TECHNICAL USER VIEW
# =====================================================

elif user_role == "Non-Technical User":

    st.header("📊 Executive AI Overview")

    st.write(f"AI Model Selected: **{best_model_name}**")

    confidence = round(val_score * 100,2)

    st.metric("AI Prediction Reliability", f"{confidence}%")

    # Executive AI Summary

    st.subheader("📊 Executive AI Summary")

    if confidence > 85:
        st.success("The AI system is performing at a very high level.")

    elif confidence > 70:
        st.info("The AI system is performing well with some room for improvement.")

    else:
        st.warning("AI predictions should be reviewed carefully.")

    # Prediction confidence

    st.subheader("📉 Prediction Confidence")

    st.progress(confidence/100)

    # Recommendations

    st.subheader("🤖 AI Recommendations")

    st.write("""
    • Continue monitoring AI predictions regularly  
    • Ensure new data follows the same format  
    • Periodically retrain the model for better accuracy  
    """)

# =====================================================
# 3️⃣ BUSINESS USER VIEW
# =====================================================

elif user_role == "Business User":

    st.header("📈 Business Intelligence Dashboard")

    if "original_df" not in st.session_state:
        st.stop()

    df = st.session_state["original_df"].copy()

    # =====================================================
    # FIX ARROW SERIALIZATION
    # =====================================================

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    numeric_cols = df.select_dtypes(include="number").columns

    # --------------------------------------------------
    # Business KPIs
    # --------------------------------------------------

    st.subheader("📈 Business KPI Insights")

    if len(numeric_cols) > 0:

        revenue_col = numeric_cols[0]

        total_revenue = df[revenue_col].sum()

        st.metric("Total Revenue", f"{round(total_revenue,2)}")

        avg_value = df[revenue_col].mean()

        st.metric("Average Transaction", round(avg_value,2))

    st.metric("Dataset Size", len(df))

    # --------------------------------------------------
    # Executive AI Summary
    # --------------------------------------------------

    st.subheader("📊 Executive AI Summary")

    st.write(f"""
    The AI model **{best_model_name}** was selected as the most reliable model
    for predicting **{target_column}**.
    """)

    reliability = round(val_score*100,2)

    st.metric("Prediction Reliability", f"{reliability}%")

    # --------------------------------------------------
    # Business Recommendations
    # --------------------------------------------------

    st.subheader("🤖 AI Business Recommendations")

    if reliability > 85:

        st.success("""
        • AI insights can be confidently used for business decisions  
        • Forecasting and planning can rely on model predictions
        """)

    elif reliability > 70:

        st.info("""
        • AI predictions are useful but should be combined with business judgement
        """)

    else:

        st.warning("""
        • AI model reliability is moderate  
        • Further data improvement recommended
        """)

    # --------------------------------------------------
    # Business Drivers
    # --------------------------------------------------

    st.subheader("🏆 Top Business Drivers")

    cat_cols = df.select_dtypes(include="object").columns

    if len(cat_cols) > 0 and len(numeric_cols) > 0:

        top = df.groupby(cat_cols[0])[numeric_cols[0]].sum().sort_values(ascending=False).head(5)

        st.dataframe(top)

    st.success("✅ Business insights generated successfully.")