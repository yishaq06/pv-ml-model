# streamlit_app.py
# Multi-page Streamlit app for PV ML forecasting

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

st.set_page_config(page_title="PV ML Forecasting", layout="wide")

MODEL_PATH = Path("pv_multioutput_model.joblib")
DATA_PATH = Path("training_dataset.parquet")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dataset Upload & Training", "PV Forecasting"],
)

# ======================================================
# PAGE 1: DATASET UPLOAD & MODEL TRAINING
# ======================================================
if page == "Dataset Upload & Training":
    st.title("Dataset Upload & Model Training")

    st.markdown(
        "Use this page to upload historical PV data and train the forecasting model."
    )

    uploaded_file = st.file_uploader(
        "Upload historical PV dataset (Excel or CSV)",
        type=["xlsx", "csv"],
    )

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        all_columns = df.columns.tolist()

        st.subheader("Feature Selection")
        input_features = st.multiselect(
            "Select input (independent) variables",
            all_columns,
        )

        target_features = st.multiselect(
            "Select target (prediction) variables",
            all_columns,
        )

        if input_features and target_features:
            X = df[input_features]
            y = df[target_features]

            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random state", value=42, step=1)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            st.subheader("Model Configuration")
            n_estimators = st.slider("Random Forest trees", 100, 1000, 300, step=50)
            max_depth = st.slider("Max depth", 3, 30, 15)

            model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "regressor",
                        MultiOutputRegressor(
                            RandomForestRegressor(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                random_state=random_state,
                                n_jobs=-1,
                            )
                        ),
                    ),
                ]
            )

            if st.button("Train & Save Model"):
                with st.spinner("Training model..."):
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metrics = []
                for i, col in enumerate(target_features):
                    metrics.append(
                        {
                            "Target": col,
                            "R2": r2_score(y_test.iloc[:, i], y_pred[:, i]),
                            "MAE": mean_absolute_error(y_test.iloc[:, i], y_pred[:, i]),
                        }
                    )

                st.subheader("Model Performance")
                st.dataframe(pd.DataFrame(metrics))

                joblib.dump(
                    {
                        "model": model,
                        "input_features": input_features,
                        "target_features": target_features,
                    },
                    MODEL_PATH,
                )

                df.to_parquet(DATA_PATH)

                st.success("Model and dataset saved successfully.")

    else:
        st.info("Please upload a dataset to begin training.")

# ======================================================
# PAGE 2: PV FORECASTING
# ======================================================
elif page == "PV Forecasting":
    st.title("PV Economic & Environmental Forecasting")

    if not MODEL_PATH.exists():
        st.warning("No trained model found. Please train a model first.")
        st.stop()

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    input_features = bundle["input_features"]
    target_features = bundle["target_features"]

    st.markdown("Enter PV project parameters below to forecast outcomes.")

    if DATA_PATH.exists():
        ref_df = pd.read_parquet(DATA_PATH)
    else:
        ref_df = None

    user_input = {}
    for col in input_features:
        default_val = (
            float(ref_df[col].mean()) if ref_df is not None else 0.0
        )
        user_input[col] = st.number_input(col, value=default_val)

    input_df = pd.DataFrame([user_input])

    if st.button("Predict Metrics"):
        predictions = model.predict(input_df)
        result_df = pd.DataFrame(predictions, columns=target_features)

        st.subheader("Predicted Results")
        st.dataframe(result_df)
