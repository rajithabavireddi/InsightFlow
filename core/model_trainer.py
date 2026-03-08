import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score


# =====================================================
# PROBLEM TYPE DETECTION
# =====================================================

def detect_problem_type(y):

    if y.dtype == "object" or y.dtype.name == "category":
        return "Classification"

    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
        return "Classification"

    return "Regression"


# =====================================================
# TRAIN + COMPARE MODELS
# =====================================================

def train_and_compare(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    problem_type = detect_problem_type(y)

    # =====================================================
    # TRAIN / VALIDATION / TEST SPLIT
    # =====================================================

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    split_info = {
        "Train": len(X_train),
        "Validation": len(X_val),
        "Test": len(X_test),
        "Total": len(df)
    }

    # =====================================================
    # PREPROCESSOR
    # =====================================================

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    # =====================================================
    # MODEL SELECTION
    # =====================================================

    if problem_type == "Classification":
        models = {
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
        scoring = "accuracy"
    else:
        models = {
            "Random Forest": RandomForestRegressor(random_state=42),
            "Linear Regression": LinearRegression()
        }
        scoring = "r2"

    results = []

    # =====================================================
    # TRAIN MODELS
    # =====================================================

    for name, model in models.items():

        start_time = time.time()

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", model)
        ])

        # Cross validation on TRAIN only
        cv_score = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring=scoring
        ).mean()

        # Fit on full training set
        pipeline.fit(X_train, y_train)

        # Evaluate on validation
        val_preds = pipeline.predict(X_val)

        if problem_type == "Classification":
            val_score = accuracy_score(y_val, val_preds)
        else:
            val_score = r2_score(y_val, val_preds)

        # Final test evaluation
        test_preds = pipeline.predict(X_test)

        if problem_type == "Classification":
            test_score = accuracy_score(y_test, test_preds)
        else:
            test_score = r2_score(y_test, test_preds)

        training_time = round(time.time() - start_time, 3)

        results.append({
            "Model": name,
            "CV Score": round(cv_score, 4),
            "Validation Score": round(val_score, 4),
            "Test Score": round(test_score, 4),
            "Training Time (sec)": training_time,
            "Pipeline": pipeline
        })

    leaderboard = sorted(results, key=lambda x: x["Test Score"], reverse=True)

    best = leaderboard[0]

    return {
        "problem_type": problem_type,
        "leaderboard": leaderboard,
        "best_model": best["Pipeline"],
        "best_model_name": best["Model"],
        "best_score": best["Test Score"],
        "metric_name": "Accuracy" if problem_type == "Classification" else "R2 Score",
        "split_info": split_info
    }