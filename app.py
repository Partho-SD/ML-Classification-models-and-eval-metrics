import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PolynomialFeatures
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import tempfile
import requests

st.set_page_config(page_title="Diabetes Classification App", layout="centered")
st.markdown("<h3 style='font-size:20px;'>Diabetes Classification Model Evaluation</h3>", unsafe_allow_html=True)

# Sidebar: Upload dataset and select model
st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# List of available model files in the GitHub repo
GITHUB_MODELS = [
    "Logistic_Regression_model.pkl",
    "Decision_Tree_model.pkl",
    "K_nearest_Neighbour_model.pkl",
    "Gaussian_Naive_Bayes_model.pkl",
    "Random_Forest_model.pkl",
    "XgBoost_model.pkl"
]
GITHUB_BASE_URL = "https://github.com/Partho-SD/ML-Classification-models-and-eval-metrics/raw/main/models/"

st.sidebar.header("2. Select Model from GitHub")
selected_model_name = st.sidebar.selectbox("Choose Model", GITHUB_MODELS)

def download_model_from_github(model_name):
    url = GITHUB_BASE_URL + model_name
    response = requests.get(url)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    else:
        st.error(f"Failed to download {model_name} from GitHub.")
        return None

selected_model_path = download_model_from_github(selected_model_name)

# model_names and model_folder logic removed as models are now downloaded from GitHub

def add_polynomial_features(X, non_binary_cols):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X[non_binary_cols])
    poly_feature_names = poly.get_feature_names_out(non_binary_cols)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names, index=X.index)
    # Remove columns that already exist in X to avoid duplication
    X_poly_df = X_poly_df.loc[:, ~X_poly_df.columns.isin(X.columns)]
    return pd.concat([X, X_poly_df], axis=1), list(X_poly_df.columns)

if uploaded_file is not None and selected_model_path:
    df = pd.read_csv(uploaded_file)
    # Remove any duplicate columns from the uploaded data
    df = df.loc[:, ~df.columns.duplicated()]

    st.header("Features and Target")
    all_columns = df.columns.tolist()
    target_col = all_columns[-1]
    st.write(f"Target Column: {target_col}")
    feature_cols = [col for col in all_columns if col != target_col]
    st.write(f"Feature Columns: {feature_cols}")

    if feature_cols and target_col:
        X = df[feature_cols].copy()
        y = df[target_col]

        # Remove any duplicate columns from X
        X = X.loc[:, ~X.columns.duplicated()]

        # Feature engineering
        # Only add engineered columns if they don't already exist
        for col in ['BMI', 'GenHlth']:
            if col in X.columns and f'{col}_log' not in X.columns:
                X[f'{col}_log'] = np.log1p(X[col])
        scaler = StandardScaler()
        for col in ['BMI_log', 'GenHlth_log']:
            if col in X.columns and f'{col}_scaled' not in X.columns:
                X[f'{col}_scaled'] = scaler.fit_transform(X[[col]])
        if 'Age' in X.columns and 'Age_squared' not in X.columns:
            X['Age_squared'] = X['Age'] ** 2
        if 'Age_squared' in X.columns and 'Age_squared_scaled' not in X.columns:
            age_scaler = StandardScaler()
            X['Age_squared_scaled'] = age_scaler.fit_transform(X[['Age_squared']])
        for col in ['Education', 'Income']:
            if col in X.columns and f'{col}_robust' not in X.columns:
                scaler_robust = RobustScaler()
                X[f'{col}_robust'] = scaler_robust.fit_transform(X[[col]])
        for col in ['MentHlth', 'PhysHlth']:
            if col in X.columns and f'{col}_minmax' not in X.columns:
                scaler_minmax = MinMaxScaler()
                X[f'{col}_minmax'] = scaler_minmax.fit_transform(X[[col]])

        # Remove any duplicate columns from X after feature engineering
        X = X.loc[:, ~X.columns.duplicated()]
        # Feature lists (use list)
        Trainlist = [
            'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
            'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
            'NoDocbcCost', 'DiffWalk', 'Sex', 'BMI_log_scaled', 'GenHlth_log_scaled',
            'Age_squared_scaled', 'Education_robust', 'Income_robust',
            'MentHlth_minmax', 'PhysHlth_minmax'
        ]
        binary_cols_trainlist = [col for col in Trainlist if set(X[col].dropna().unique()).issubset({0.0, 1.0})]
        non_binary_cols_trainlist = [col for col in Trainlist if col not in binary_cols_trainlist]

        # Polynomial features (avoid duplicate columns)
        X, poly_feature_names = add_polynomial_features(X, non_binary_cols_trainlist)

        # Remove any duplicate columns from X after polynomial features
        X = X.loc[:, ~X.columns.duplicated()]

        # Load model after all transformations
        model = joblib.load(selected_model_path)

        # Choose feature list based on model name
        if selected_model_name == "Logistic_Regression_model.pkl":
            feature_list = Trainlist + list(poly_feature_names)
        else:
            feature_list = Trainlist

        # Remove duplicates from feature_list while preserving order
        seen = set()
        feature_list = [x for x in feature_list if not (x in seen or seen.add(x))]
        # Ensure all features exist in X
        feature_list = [f for f in feature_list if f in X.columns]

        # Predict
        y_pred = model.predict(X[feature_list])
        y_prob = model.predict_proba(X[feature_list])[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics and Confusion Matrix side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4 style='font-size:16px;'>Evaluation Metrics</h4>", unsafe_allow_html=True)
            metrics = {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred, zero_division=0),
            "Recall": recall_score(y, y_pred, zero_division=0),
            "F1 Score": f1_score(y, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y, y_pred),
            "AUC": roc_auc_score(y, y_prob) if y_prob is not None else np.nan
            }
            st.table(pd.DataFrame(metrics, index=["Score"]).T)

        with col2:
            st.markdown("<h4 style='font-size:16px;'>Confusion Matrix</h4>", unsafe_allow_html=True)
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks(np.arange(cm.shape[1]))
            ax.set_yticks(np.arange(cm.shape[0]))
            ax.set_xticklabels(["0", "1"])
            ax.set_yticklabels(["0", "1"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
            fig.colorbar(im)
            st.pyplot(fig)
    else:
        st.info("Please select at least one feature and a target column.")
elif uploaded_file is None:
    st.info("Please upload a CSV file to begin.")
elif not selected_model_path:
    st.info("Please upload at least one trained model (.pkl) file.")