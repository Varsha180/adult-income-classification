"""Streamlit application for Adult Census Income Classification."""

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

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

EXPECTED_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

st.set_page_config(
    page_title="Adult Income Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Adult Census Income Classification")

st.write(
    "Upload your test dataset, select a machine learning model, "
    "and evaluate its classification performance."
)


# --------------------------------------------------
# 2. Load Saved Model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# 3. Target Encoding
# --------------------------------------------------

def encode_target(values):

    if pd.api.types.is_numeric_dtype(values):
        encoded = values.astype(int)

        if not encoded.isin([0, 1]).all():
            raise ValueError(
                "Numeric income labels must contain only 0 and 1."
            )

        return encoded

    mapping = {
        "<=50K": 0,
        ">50K": 1,
        "<=50K.": 0,
        ">50K.": 1
    }

    encoded = (
        values.astype(str)
        .str.strip()
        .map(mapping)
    )

    if encoded.isna().any():
        raise ValueError(
            "Invalid income labels. Expected <=50K or >50K."
        )

    return encoded.astype(int)


# --------------------------------------------------
# 4. Dataset Upload
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test dataset (CSV)",
    type=["csv"],
    help="Upload only the test dataset. Do not upload training data."
)

if uploaded_file is None:
    st.info("Please upload your test dataset to continue.")
    st.stop()

try:

    # Read CSV without automatically converting '?' to NaN
    data = pd.read_csv(
        uploaded_file,
        keep_default_na=False
    )

    # Remove accidental spaces from column names
    data.columns = data.columns.str.strip()

    # Validate dataset columns
    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in data.columns
    ]

    extra_columns = [
        col for col in data.columns
        if col not in EXPECTED_COLUMNS
    ]

    if missing_columns:
        st.error(
            f"Missing required columns: {missing_columns}"
        )
        st.stop()

    if extra_columns:
        st.error(
            f"Unexpected columns found: {extra_columns}"
        )
        st.stop()

    # Arrange columns in expected order
    data = data[EXPECTED_COLUMNS]

    if data.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    # --------------------------------------------------
    # 5. Missing Value Analysis
    # --------------------------------------------------

    st.subheader("Dataset Validation")

    # Actual null values
    null_counts = data.isnull().sum()

    # Blank and whitespace-only values
    blank_counts = (
        data.astype("string")
        .apply(
            lambda col: col.str.strip().eq("").sum()
        )
    )

    # '?' values
    question_counts = (
        data.astype("string")
        .apply(
            lambda col: col.str.strip().eq("?").sum()
        )
    )

    missing_summary = pd.DataFrame({
        "Null Values": null_counts,
        "Blank Values": blank_counts,
        "'?' Values": question_counts
    })

    st.dataframe(
        missing_summary,
        use_container_width=True
    )

    total_missing = (
        null_counts.sum()
        + blank_counts.sum()
        + question_counts.sum()
    )

    if total_missing > 0:

        st.warning(
            f"Dataset contains {total_missing} missing or "
            "invalid entries. Please check the dataset."
        )

        st.stop()

    st.success("Dataset uploaded and validated successfully!")

    # Dataset information
    col1, col2 = st.columns(2)

    col1.metric("Total Test Records", len(data))
    col2.metric("Total Features", len(EXPECTED_COLUMNS) - 1)

    with st.expander("View Uploaded Dataset"):
        st.dataframe(data, use_container_width=True)

except Exception as e:

    st.error(f"Error reading dataset: {e}")
    st.stop()


# --------------------------------------------------
# 6. Model Selection
# --------------------------------------------------

st.divider()

st.header("2. Select Classification Model")

model_name = st.selectbox(
    "Choose a trained classifier",
    options=list(MODEL_FILES.keys())
)


# --------------------------------------------------
# 7. Model Evaluation
# --------------------------------------------------

if st.button("Evaluate Model", type="primary"):

    model_path = MODEL_DIR / MODEL_FILES[model_name]

    if not model_path.exists():

        st.error(
            f"Model file not found: {model_path.name}. "
            "Please run train_models.py first."
        )

        st.stop()

    try:

        model = load_model(str(model_path))

        # Separate input features and target
        X_test = data.drop(columns=["income"])
        y_test = encode_target(data["income"])

        # Generate predictions
        y_pred = model.predict(X_test)

        # Ensure predictions are binary
        y_pred = pd.Series(y_pred).astype(int)

        if not y_pred.isin([0, 1]).all():
            raise ValueError(
                "The model returned unexpected class labels."
            )

        # --------------------------------------------------
        # 8. Evaluation Metrics
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        st.divider()

        st.header("3. Evaluation Metrics")

        st.caption(f"Selected Model: {model_name}")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1-Score", f"{f1:.4f}")

        # --------------------------------------------------
        # 9. Confusion Matrix
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
            columns=[
                "Predicted <=50K",
                "Predicted >50K"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        # --------------------------------------------------
        # 10. Classification Report
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

        st.error(
            f"Model evaluation failed: {e}"
        )