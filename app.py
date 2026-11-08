"""Streamlit interface for the five saved Adult Income classifiers."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

st.set_page_config(page_title="Adult Income Classifier", page_icon="📊", layout="wide")
st.title("Adult Census Income Classification")
st.write(
    "Select one of the five trained classifiers and a row from the held-out "
    "test dataset to view its prediction."
)

@st.cache_data
def load_test_data():
    return pd.read_csv(ROOT / "test_data.csv")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

test_path = ROOT / "test_data.csv"
if not test_path.exists():
    st.error("test_data.csv is missing. Run `python train_models.py` first.")
    st.stop()

data = load_test_data()
target_column = "income"
feature_data = data.drop(columns=[target_column])

model_name = st.selectbox("Choose classifier", list(MODEL_FILES.keys()))
row_number = st.number_input(
    "Select test record number",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)

selected_row = feature_data.iloc[[int(row_number)]]
actual_value = data.iloc[int(row_number)][target_column]

with st.expander("View selected input record"):
    st.dataframe(selected_row, use_container_width=True)

if st.button("Predict income class", type="primary"):
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        st.error(f"Model file not found: {model_path.name}. Run train_models.py.")
        st.stop()

    model = load_model(model_path)
    prediction = int(model.predict(selected_row)[0])
    predicted_label = ">50K" if prediction == 1 else "<=50K"

    st.subheader("Prediction result")
    col1, col2 = st.columns(2)
    col1.metric("Predicted income class", predicted_label)
    col2.metric("Actual income class (test label)", actual_value)
    if predicted_label == actual_value:
        st.success("The prediction matches the actual test label.")
    else:
        st.warning("The prediction does not match the actual test label.")
