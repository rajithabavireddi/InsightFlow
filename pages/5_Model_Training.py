import streamlit as st

# ---------------------------------------------------
# Login Protection
# ---------------------------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please login first.")
    st.stop()

if "user_role" not in st.session_state:
    st.warning("⚠ Role not found. Please login again.")
    st.stop()

role = st.session_state.user_role

st.sidebar.subheader("👤 Current Role")
st.sidebar.info(role)

import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import accuracy_score, r2_score, confusion_matrix

st.title("🤖 AutoML Model Training & Leaderboard")

# ---------------------------------------------------
# Check Dataset
# ---------------------------------------------------
if "cleaned_df" not in st.session_state:
    st.warning("⚠ Please clean the dataset first.")
    st.stop()

df = st.session_state.cleaned_df.copy()

# ---------------------------------------------------
# Target Selection
# ---------------------------------------------------
st.subheader("📌 Select Target Column")

target_column = st.selectbox("Choose target column", df.columns)

# ---------------------------------------------------
# Train Button
# ---------------------------------------------------
if st.button("🚀 Train AutoML Models"):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ---------------------------------------------------
    # Detect Problem Type
    # ---------------------------------------------------
    if y.dtype == "object" or y.dtype == "bool" or y.nunique() <= 15:
        problem_type = "classification"
        metric_name = "Accuracy"
    else:
        problem_type = "regression"
        metric_name = "R2 Score"

    st.info(f"Detected Problem Type: **{problem_type.upper()}**")

    # ---------------------------------------------------
    # Imbalance Detection
    # ---------------------------------------------------
    if problem_type == "classification":

        class_distribution = y.value_counts(normalize=True)

        if class_distribution.max() > 0.85:
            st.warning(
                "⚠ Dataset appears to be highly imbalanced.\n"
                "Model predictions may be biased."
            )

    # ---------------------------------------------------
    # Remove Rare Classes
    # ---------------------------------------------------
    if problem_type == "classification":

        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < 2].index

        if len(rare_classes) > 0:

            st.warning("⚠ Removing classes with less than 2 samples")

            df = df[~df[target_column].isin(rare_classes)]

            X = df.drop(columns=[target_column])
            y = df[target_column]

    # ---------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------
    numeric_features = X.select_dtypes(include=["number"]).columns
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # ---------------------------------------------------
    # Train Test Split
    # ---------------------------------------------------
    stratify_option = y if problem_type == "classification" else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_option
    )

    st.write("### 📊 Dataset Split")
    st.write(f"Training Size: {len(X_train)}")
    st.write(f"Test Size: {len(X_test)}")

    # ---------------------------------------------------
    # AutoML Models
    # ---------------------------------------------------
    if problem_type == "classification":

        models = {

            "Logistic Regression": LogisticRegression(max_iter=2000),

            "Random Forest":
            RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                random_state=42
            ),

            "Decision Tree":
            DecisionTreeClassifier(),

            "Gradient Boosting":
            GradientBoostingClassifier(),

            "Support Vector Machine":
            SVC(probability=True),

            "Extra Trees":
            ExtraTreesClassifier()
        }

    else:

        models = {

            "Linear Regression": LinearRegression(),

            "Ridge Regression": Ridge(),

            "Random Forest":
            RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                random_state=42
            ),

            "Decision Tree":
            DecisionTreeRegressor(),

            "Gradient Boosting":
            GradientBoostingRegressor(),

            "Support Vector Regressor":
            SVR(),

            "Extra Trees":
            ExtraTreesRegressor()
        }

    # ---------------------------------------------------
    # Train Models
    # ---------------------------------------------------
    results = []
    trained_pipelines = []

    progress = st.progress(0)

    for i, (name, model) in enumerate(models.items()):

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        start_time = time.time()

        pipeline.fit(X_train, y_train)

        train_time = time.time() - start_time

        y_pred_test = pipeline.predict(X_test)
        y_pred_train = pipeline.predict(X_train)

        if problem_type == "classification":

            train_score = accuracy_score(y_train, y_pred_train)
            test_score = accuracy_score(y_test, y_pred_test)

        else:

            train_score = r2_score(y_train, y_pred_train)
            test_score = r2_score(y_test, y_pred_test)

        results.append({
            "Model": name,
            "Train Score": round(train_score, 4),
            "Test Score": round(test_score, 4),
            metric_name: round(test_score, 4),
            "Training Time": round(train_time, 3)
        })

        trained_pipelines.append((name, pipeline))

        progress.progress((i + 1) / len(models))

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------
    # Leaderboard
    # ---------------------------------------------------
    results_df = results_df.sort_values(by=metric_name, ascending=False)

    st.subheader("🏆 AutoML Leaderboard")
    st.dataframe(results_df)

    # ---------------------------------------------------
    # Performance Chart
    # ---------------------------------------------------
    st.subheader("📊 Model Performance")

    st.bar_chart(
        results_df.set_index("Model")[metric_name]
    )

    # ---------------------------------------------------
    # Best Model
    # ---------------------------------------------------
    best_model_name = results_df.iloc[0]["Model"]

    for name, pipe in trained_pipelines:
        if name == best_model_name:
            best_pipeline = pipe

    st.success(f"🏆 Best Model: {best_model_name}")

    # ---------------------------------------------------
    # Confusion Matrix (ROLE BASED)
    # ---------------------------------------------------
    if problem_type == "classification" and role in ["Data Analyst", "Data Scientist"]:

        y_pred = best_pipeline.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        st.subheader("🧩 Confusion Matrix")

        fig, ax = plt.subplots()

        ax.imshow(cm)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

    # ---------------------------------------------------
    # Cross Validation
    # ---------------------------------------------------
    try:

        cv_scores = cross_val_score(
            best_pipeline,
            X,
            y,
            cv=5
        )

        st.subheader("🔁 Cross Validation Score")

        st.write(
            f"Average CV Score: {round(cv_scores.mean(),4)}"
        )

    except Exception as e:

        st.warning(f"Cross validation skipped: {e}")

    # ---------------------------------------------------
    # Role Based Output
    # ---------------------------------------------------
    if role == "Business User":

        best_score = results_df.iloc[0][metric_name]

        st.info(
            f"📈 AI model achieved {round(best_score*100,2)}% performance.\n"
            "This can help improve business forecasting."
        )

    elif role == "Non Technical User":

        st.info(
            "🤖 AI has successfully trained a prediction model."
        )

    elif role == "Data Analyst":

        st.info(
            "📊 Model results are ready for further analysis."
        )

    elif role == "Data Scientist":

        st.info(
            "🔬 Advanced ML evaluation completed."
        )

    # ---------------------------------------------------
    # Save Results
    # ---------------------------------------------------
    st.session_state.results_df = results_df
    st.session_state.best_model_name = best_model_name
    st.session_state.problem_type = problem_type
    st.session_state.target_column = target_column
    st.session_state.cleaned_df = df
    st.session_state.model_pipeline = best_pipeline

    st.success("✅ AutoML training completed and saved!")