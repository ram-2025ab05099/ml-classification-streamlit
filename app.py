import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer ML App",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Breast Cancer Classification App")
st.markdown("Upload test data and evaluate multiple Machine Learning models.")

# --------------------------------------------------
# Load Models and Preprocessing Objects
# --------------------------------------------------
@st.cache_resource
def load_resources():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }
    scaler = joblib.load("model/scaler.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    return models, scaler, feature_names

models, scaler, feature_names = load_resources()

# --------------------------------------------------
# Sidebar - Model Selection
# --------------------------------------------------
st.sidebar.header("⚙ Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose a Model",
    list(models.keys())
)
model = models[selected_model_name]

# --------------------------------------------------
# Dataset Upload (Requirement A)
# --------------------------------------------------
st.header("📂 Upload Test Dataset (CSV)")

uploaded_file = st.file_uploader(
    "Upload CSV file (must contain same features as training data + target column)",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Dataset Preview")
    st.dataframe(data.head())

    # Check required columns
    if "target" not in data.columns:
        st.error("Uploaded dataset must contain a 'target' column.")
    else:
        X_test = data[feature_names]
        y_test = data["target"]

        # Scale if required
        if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # Evaluation Metrics (Requirement C)
        # --------------------------------------------------
        st.header("📈 Evaluation Metrics")

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
            "Value": [accuracy, auc, precision, recall, f1, mcc]
        })

        st.dataframe(metrics_df)

        # --------------------------------------------------
        # Confusion Matrix (Requirement D)
        # --------------------------------------------------
        st.header("🔍 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Malignant", "Benign"],
                    yticklabels=["Malignant", "Benign"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # Optional: Classification Report
        st.header("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

else:
    st.info("Please upload a CSV file to evaluate the selected model.")
