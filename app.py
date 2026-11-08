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

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

st.set_page_config(
    page_title="Adult Income Classification",
    page_icon="📊",
    layout="wide"
)

st.title("Adult Census Income Classification")

st.write(
    "Upload your test dataset, select a trained machine learning "
    "classifier, and evaluate its performance."
)


# --------------------------------------------------
# 2. Load Saved Model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# 3. Encode Income Target
# --------------------------------------------------

def encode_target(values):

    mapping = {
        "<=50K": 0,
        ">50K": 1,
        "<=50K.": 0,
        ">50K.": 1,
        "0": 0,
        "1": 1
    }

    return (
        values.astype(str)
        .str.strip()
        .map(mapping)
    )


# --------------------------------------------------
# 4. Dataset Upload
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"],
    help="Upload only the test dataset. Training data is not required."
)

if uploaded_file is None:
    st.info("Please upload your test dataset to continue.")
    st.stop()


try:

    # Read CSV
    data = pd.read_csv(
        uploaded_file,
        keep_default_na=False
    )

    # Remove unwanted spaces from column names
    data.columns = data.columns.str.strip()

    # Validate columns
    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in data.columns
    ]

    extra_columns = [
        col for col in data.columns
        if col not in EXPECTED_COLUMNS
    ]

    if missing_columns or extra_columns:

        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")

        if extra_columns:
            st.error(f"Unexpected columns: {extra_columns}")

        st.stop()

    # Maintain expected column order
    data = data[EXPECTED_COLUMNS].copy()

    if data.empty:
        st.error("The uploaded dataset is empty.")
        st.stop()

    # Remove surrounding whitespace from string values
    for column in data.select_dtypes(include=["object"]).columns:
        data[column] = data[column].str.strip()

    # --------------------------------------------------
    # 5. Identify Invalid Records
    # --------------------------------------------------

    # Check actual null values
    null_mask = data.isnull().any(axis=1)

    # Check blank values
    blank_mask = (
        data.astype("string")
        .apply(lambda col: col.str.strip().eq(""))
        .any(axis=1)
    )

    # Check '?' placeholders
    question_mask = (
        data.astype("string")
        .apply(lambda col: col.str.strip().eq("?"))
        .any(axis=1)
    )

    # Check invalid numeric values
    numeric_invalid_mask = pd.Series(
        False,
        index=data.index
    )

    for column in NUMERIC_COLUMNS:

        converted = pd.to_numeric(
            data[column],
            errors="coerce"
        )

        numeric_invalid_mask |= converted.isna()

        data[column] = converted

    # Validate income labels
    encoded_income = encode_target(data["income"])

    invalid_target_mask = encoded_income.isna()

    # Combine all invalid conditions
    invalid_mask = (
        null_mask
        | blank_mask
        | question_mask
        | numeric_invalid_mask
        | invalid_target_mask
    )

    # --------------------------------------------------
    # 6. Separate Valid and Invalid Records
    # --------------------------------------------------

    invalid_records = data[invalid_mask].copy()

    valid_data = data[~invalid_mask].copy()

    # --------------------------------------------------
    # 7. Display Dataset Validation
    # --------------------------------------------------

    st.divider()

    st.subheader("2. Dataset Validation")

    total_records = len(data)
    valid_records_count = len(valid_data)
    excluded_records_count = len(invalid_records)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        total_records
    )

    col2.metric(
        "Valid Records",
        valid_records_count
    )

    col3.metric(
        "Excluded Records",
        excluded_records_count
    )

    if excluded_records_count > 0:

        st.warning(
            f"{excluded_records_count} records contain missing "
            "or invalid values. These records will be excluded "
            "from model evaluation."
        )

        with st.expander("View Excluded Records"):

            st.dataframe(
                invalid_records,
                use_container_width=True
            )

    else:

        st.success(
            "No invalid records found. All records are valid."
        )

    if valid_data.empty:

        st.error(
            "No valid records available for model evaluation."
        )

        st.stop()

    st.success(
        f"Proceeding with {valid_records_count} valid records."
    )

    # Preview valid dataset
    with st.expander("View Valid Test Dataset"):

        st.dataframe(
            valid_data,
            use_container_width=True
        )

    # Prepare target and features
    X_test = valid_data.drop(columns=["income"])

    y_test = (
        encode_target(valid_data["income"])
        .astype(int)
    )

except Exception as e:

    st.error(f"Dataset processing failed: {e}")
    st.stop()


# --------------------------------------------------
# 8. Model Selection
# --------------------------------------------------

st.divider()

st.header("3. Model Selection")

model_name = st.selectbox(
    "Choose a classification model",
    options=list(MODEL_FILES.keys())
)


# --------------------------------------------------
# 9. Model Evaluation
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

        # Load selected model
        model = load_model(str(model_path))

        # Generate predictions
        y_pred = model.predict(X_test)

        # Convert predictions to binary labels
        y_pred = encode_target(
            pd.Series(y_pred)
        )

        if y_pred.isna().any():
            raise ValueError(
                "The model returned invalid prediction labels."
            )

        y_pred = y_pred.astype(int)

        # --------------------------------------------------
        # 10. Evaluation Metrics
        # --------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

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

        st.header("4. Model Evaluation Metrics")

        st.write(f"**Selected Model:** {model_name}")

        st.write(
            f"**Records evaluated:** {len(X_test)}"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col3.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col4.metric(
            "F1-Score",
            f"{f1:.4f}"
        )

        # --------------------------------------------------
        # 11. Confusion Matrix
        # --------------------------------------------------

        st.divider()

        st.header("5. Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual <=50K",
                "Actual >50K"
            ],
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
        # 12. Classification Report
        # --------------------------------------------------

        st.header("6. Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "<=50K",
                ">50K"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # --------------------------------------------------
        # 13. Final Status
        # --------------------------------------------------

        st.success(
            f"{model_name} evaluation completed successfully!"
        )

        if excluded_records_count > 0:

            st.info(
                f"Note: {excluded_records_count} invalid records "
                "were excluded. Metrics were calculated using "
                f"the remaining {valid_records_count} records."
            )

    except Exception as e:

        st.error(
            f"Model evaluation failed: {e}"
        )