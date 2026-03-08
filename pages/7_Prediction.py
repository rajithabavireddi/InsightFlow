import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# =========================================================
# LOGIN CHECK
# =========================================================

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()

st.title("🔮 Smart Prediction")

# =========================================================
# CHECK IF MODEL IS TRAINED
# =========================================================

required_keys = ["model_pipeline", "target_column", "cleaned_df"]

for key in required_keys:
    if key not in st.session_state:
        st.warning("⚠ Please train a model first before making predictions.")
        st.stop()

pipeline = st.session_state.model_pipeline
target_column = st.session_state.target_column
raw_df = st.session_state.cleaned_df
best_model_name = st.session_state.get("best_model_name", "Trained Model")

user_role = st.session_state.get("user_role", "viewer")

st.success(f"✅ Model Ready: {best_model_name}")

# =========================================================
# FEATURE COLUMNS
# =========================================================

X_columns = raw_df.drop(columns=[target_column]).columns.tolist()

# =========================================================
# USER INPUT FORM
# =========================================================

st.subheader("📥 Enter Feature Values")

user_input = {}

for col in X_columns:

    # -------------------------
    # NUMERIC COLUMN
    # -------------------------
    if pd.api.types.is_numeric_dtype(raw_df[col]):

        min_val = float(raw_df[col].min())
        max_val = float(raw_df[col].max())
        mean_val = float(raw_df[col].mean())

        user_input[col] = st.slider(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

    # -------------------------
    # CATEGORICAL COLUMN
    # -------------------------
    else:

        options = sorted(raw_df[col].dropna().astype(str).unique().tolist())

        user_input[col] = st.selectbox(
            label=col,
            options=options
        )

# =========================================================
# PREDICTION
# =========================================================

if st.button("🚀 Predict"):

    input_df = pd.DataFrame([user_input])

    prediction = pipeline.predict(input_df)

    st.success("📌 AI Prediction Result")

    predicted_output = prediction[0]

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    if isinstance(predicted_output, (str, np.str_, object)):

        st.markdown(f"""
        ### 🎯 Predicted Outcome  
        **{target_column} = {predicted_output}**
        """)

        model = pipeline.named_steps.get("model", None)

        if model and hasattr(model, "predict_proba"):

            probabilities = pipeline.predict_proba(input_df)[0]

            prob_df = pd.DataFrame({
                "Class": model.classes_,
                "Probability": probabilities
            })

            confidence = round(np.max(probabilities) * 100, 2)

            st.write(f"Confidence Level: **{confidence}%**")

            st.bar_chart(prob_df.set_index("Class"))

    # =====================================================
    # REGRESSION
    # =====================================================

    else:

        predicted_value = float(predicted_output)

        st.markdown(f"""
        ### 📊 Predicted Value  
        **{target_column} = {round(predicted_value, 4)}**
        """)

        st.info("This value is calculated from patterns learned during training.")

    # =====================================================
    # SHAP EXPLAINABILITY
    # ONLY FOR ANALYST & DATA SCIENTIST
    # =====================================================

    if user_role in ["data_analyst", "data_scientist"]:

        st.markdown("---")
        st.subheader("🧠 AI Explanation (SHAP)")

        try:

            # get model from pipeline
            model = pipeline.named_steps["model"]

            # training data for background
            X_train = raw_df.drop(columns=[target_column])

            # create explainer
            explainer = shap.Explainer(model, X_train)

            shap_values = explainer(input_df)

            # =================================================
            # WATERFALL PLOT
            # =================================================

            st.write("### Feature Contribution")

            fig, ax = plt.subplots()

            shap.plots.waterfall(shap_values[0], show=False)

            st.pyplot(fig)

            # =================================================
            # FORCE PLOT
            # =================================================

            st.write("### Prediction Breakdown")

            force_plot = shap.plots.force(
                shap_values.base_values[0],
                shap_values.values[0],
                input_df.iloc[0],
                matplotlib=True
            )

            st.pyplot(force_plot)

        except Exception as e:

            st.warning("SHAP explanation not available for this model.")