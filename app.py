"""Streamlit interface for the five saved Adult Income classifiers."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# Page configuration
st.set_page_config(
    page_title="Adult Income Classifier",
    page_icon="📊",
    layout="wide"
)

st.title("Adult Census Income Classification")
st.write(
    "Upload your test dataset, select a trained classification model, "
    "and evaluate its performance."
)

# Load saved model
@st.cache_resource
def load_model(path):
    return joblib.load(path)


# Convert target labels into binary values
def encode_target(values):
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(int)

    mapping = {
        "<=50K": 0,
        ">50K": 1,
        "<=50K.": 0,
        ">50K.": 1
    }

    encoded = values.astype(str).str.strip().map(mapping)

    if encoded.isna().any():
        raise ValueError(
            "Invalid income labels. Expected '<=50K' or '>50K'."
        )

    return encoded.astype(int)


# --------------------------------------------------
# 1. Dataset Upload
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV format)",
    type=["csv"],
    help="Upload only the test dataset containing the income target column."
)

if uploaded_file is None:
    st.info("Please upload your test dataset to continue.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)

    if "income" not in data.columns:
        st.error("The uploaded dataset must contain an 'income' column.")
        st.stop()

    if data.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    if data.isnull().any().any():
        st.error("The uploaded dataset contains missing values.")
        st.stop()

    st.success("Test dataset uploaded successfully!")

    st.write("Dataset Preview")
    st.dataframe(data.head(), use_container_width=True)

    st.write(f"Total test records: **{len(data)}**")

except Exception as e:
    st.error(f"Unable to read the uploaded CSV: {e}")
    st.stop()


# --------------------------------------------------
# 2. Model Selection
# --------------------------------------------------

st.header("2. Select Classification Model")

model_name = st.selectbox(
    "Choose a classifier",
    list(MODEL_FILES.keys())
)


# --------------------------------------------------
# 3. Model Evaluation
# --------------------------------------------------

if st.button("Evaluate Model", type="primary"):

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():
        st.error(
            f"Model file not found: {model_path.name}. "
            "Run train_models.py first."
        )
        st.stop()

    try:
        model = load_model(str(model_path))

        # Separate features and actual target
        X_test = data.drop(columns=["income"])
        y_test = encode_target(data["income"])

        # Generate predictions
        y_pred = model.predict(X_test)

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        st.divider()

        st.header("3. Model Evaluation Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1-Score", f"{f1:.4f}")

        # --------------------------------------------------
        # 4. Confusion Matrix
        # --------------------------------------------------

        st.divider()

        st.header("4. Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual <=50K", "Actual >50K"],
            columns=["Predicted <=50K", "Predicted >50K"]
        )

        st.dataframe(cm_df, use_container_width=True)

        # --------------------------------------------------
        # 5. Classification Report
        # --------------------------------------------------

        st.header("5. Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["<=50K", ">50K"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        st.success(
            f"{model_name} evaluation completed successfully!"
        )

    except Exception as e:
        st.error(f"Model evaluation failed: {e}")