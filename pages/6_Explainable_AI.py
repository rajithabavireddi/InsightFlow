import streamlit as st
from core.arrow_fix import make_arrow_compatible

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap

st.title("🧠 Explainable AI")

# -------------------------
# Check model availability
# -------------------------
if "model_pipeline" not in st.session_state:
    st.warning("⚠ Please train the model first.")
    st.stop()

pipeline = st.session_state.model_pipeline
problem_type = st.session_state.problem_type

# -------------------------
# Fix Arrow serialization
# -------------------------
df = st.session_state.cleaned_df.copy()

df = make_arrow_compatible(df)
st.session_state.cleaned_df = df

st.success("✅ Trained Model Loaded")

# -------------------------
# Extract Model + Preprocessor
# -------------------------

model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["preprocessor"]

target_column = st.session_state.target_column

X = df.drop(columns=[target_column])

# -------------------------
# Transform features
# -------------------------

X_transformed = preprocessor.transform(X)

if not isinstance(X_transformed, np.ndarray):
    X_transformed = X_transformed.toarray()

# Try getting feature names
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    feature_names = np.array([f"Feature_{i}" for i in range(X_transformed.shape[1])])


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

st.subheader("📊 Feature Importance")

try:

    # Tree Models
    if hasattr(model, "feature_importances_"):

        importances = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(importance_df, width="stretch")

        fig, ax = plt.subplots()

        ax.barh(
            importance_df["Feature"][:15],
            importance_df["Importance"][:15]
        )

        ax.invert_yaxis()
        ax.set_title("Top 15 Feature Importances")

        st.pyplot(fig)

    # Linear Models
    elif hasattr(model, "coef_"):

        coefs = model.coef_

        if len(coefs.shape) > 1:
            coefs = coefs[0]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefs
        })

        importance_df["Abs"] = importance_df["Coefficient"].abs()

        importance_df = importance_df.sort_values(
            by="Abs",
            ascending=False
        )

        st.dataframe(
            importance_df.drop(columns=["Abs"]),
            width="stretch"
        )

        fig, ax = plt.subplots()

        ax.barh(
            importance_df["Feature"][:15],
            importance_df["Coefficient"][:15]
        )

        ax.invert_yaxis()
        ax.set_title("Top 15 Coefficients")

        st.pyplot(fig)

    else:

        st.info(
            "ℹ️ This model does not provide built-in feature importance. "
            "Use SHAP Explainability below."
        )

except Exception as e:

    st.error(f"Error generating explanation: {e}")


# ==================================================
# SHAP EXPLAINABILITY
# ==================================================

if "user_role" in st.session_state and st.session_state.user_role in ["Data Analyst", "Data Scientist"]:

    st.divider()
    st.subheader("🧠 SHAP Explainability")

    try:

        # ----------------------------------
        # Speed optimization
        # ----------------------------------

        if len(X_transformed) > 200:
            idx = np.random.choice(len(X_transformed), 200, replace=False)
            X_sample = X_transformed[idx]
        else:
            X_sample = X_transformed

        # ==================================
        # SHAP EXPLAINER SELECTION
        # ==================================

        with st.spinner("Generating SHAP explanations..."):

            if hasattr(model, "feature_importances_"):

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)

            elif hasattr(model, "coef_"):

                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)

            else:

                background = shap.sample(X_sample, min(50, len(X_sample)))

                explainer = shap.KernelExplainer(model.predict, background)
                shap_values = explainer.shap_values(X_sample[:50])

        # ----------------------------------
        # Fix shape
        # ----------------------------------

        if isinstance(shap_values, list):
            shap_array = shap_values[0]
        else:
            shap_array = shap_values

        shap_array = np.array(shap_array)

        if len(shap_array.shape) == 3:
            shap_array = shap_array[:, :, 0]

        # ==================================
        # DATA ANALYST VIEW
        # ==================================

        if st.session_state.user_role == "Data Analyst":

            st.write("### 📊 Global Feature Impact")

            fig = plt.figure()

            shap.summary_plot(
                shap_array,
                X_sample[:len(shap_array)],
                plot_type="bar",
                feature_names=feature_names,
                show=False
            )

            st.pyplot(fig)

        # ==================================
        # DATA SCIENTIST VIEW
        # ==================================

        if st.session_state.user_role == "Data Scientist":

            st.write("### 🌍 SHAP Summary Plot")

            fig1 = plt.figure()

            shap.summary_plot(
                shap_array,
                X_sample[:len(shap_array)],
                feature_names=feature_names,
                show=False
            )

            st.pyplot(fig1)

            st.write("### 📊 SHAP Feature Importance")

            fig2 = plt.figure()

            shap.summary_plot(
                shap_array,
                X_sample[:len(shap_array)],
                plot_type="bar",
                feature_names=feature_names,
                show=False
            )

            st.pyplot(fig2)

    except Exception as e:

        st.error(f"SHAP could not be generated: {e}")