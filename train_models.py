"""
Adult Census Income Classification: tuned training, evaluation and graphs.

Run from this project folder:
    python train_models.py

Expected dataset location:
    project-folder/
        train_models.py
        Dataset/
            adult.data
            adult.test

The official adult.data file is used for training and cross-validation.
The official adult.test file is kept untouched for final evaluation.
All preprocessing is inside each Pipeline to prevent preprocessing leakage.
"""
from pathlib import Path
import warnings
import html

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# ---------------------------------------------------------
# 1. Project paths and configuration
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "Dataset"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CV_FOLDS = 3
SCORING_METRIC = "f1"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

MODEL_FILE_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}


# ---------------------------------------------------------
# 2. Load the original UCI Adult files
# ---------------------------------------------------------
def clean_target(target):
    """Normalize target labels in adult.data and adult.test."""
    target = target.astype(str).str.strip()
    target = target.str.replace(".", "", regex=False)
    return target


def load_dataset():
    """
    Use adult.data as training data and adult.test as the untouched test data.
    Do not combine the two files and randomly split them; the official test
    partition should remain separate during hyperparameter tuning.
    """
    train_path = DATASET_DIR / "adult.data"
    test_path = DATASET_DIR / "adult.test"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Dataset files not found. Expected adult.data and adult.test "
            f"inside: {DATASET_DIR}"
        )

    train_df = pd.read_csv(
        train_path,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
        comment="|"
    )

    test_df = pd.read_csv(
        test_path,
        names=COLUMNS,
        skiprows=1,
        skipinitialspace=True,
        na_values="?",
        comment="|"
    )

    # Remove blank or malformed rows, if any.
    train_df = train_df.dropna(subset=["income"]).copy()
    test_df = test_df.dropna(subset=["income"]).copy()

    train_df["income"] = clean_target(train_df["income"])
    test_df["income"] = clean_target(test_df["income"])

    target_mapping = {"<=50K": 0, ">50K": 1}
    y_train = train_df["income"].map(target_mapping)
    y_test = test_df["income"].map(target_mapping)

    # Fail early if an unexpected target label exists.
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Unexpected target labels found in the income column.")

    X_train = train_df.drop(columns=["income"])
    X_test = test_df.drop(columns=["income"])

    # Ensure the test columns follow the same order as training columns.
    X_test = X_test[X_train.columns]

    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows : {len(X_test):,}")
    print(f"Features     : {X_train.shape[1]}")

    return X_train, X_test, y_train.astype(int), y_test.astype(int)


# ---------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------
def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])


# ---------------------------------------------------------
# 4. Define the five required classifiers and search spaces
# ---------------------------------------------------------
def build_models_and_parameters(preprocessor):
    models = {
        "Logistic Regression": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", LogisticRegression(
                max_iter=2500,
                solver="liblinear",
                random_state=RANDOM_STATE
            ))
        ]),

        "Decision Tree": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", DecisionTreeClassifier(
                random_state=RANDOM_STATE
            ))
        ]),

        "KNN": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", KNeighborsClassifier(
                n_jobs=-1
            ))
        ]),

        "Naive Bayes": Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", GaussianNB())
        ]),

        "Support Vector Machine": Pipeline([
            # Keep one-hot features sparse to reduce SVM memory use.
            ("preprocessor", clone(preprocessor).set_params(
                categorical__onehot__sparse_output=True,
                sparse_threshold=1.0
            )),
            ("classifier", SGDClassifier(
                loss="hinge",              # Linear SVM objective
                penalty="l2",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=1000,
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }

    # These grids are intentionally bounded to keep execution practical.
    # GridSearchCV selects parameters using training data only.
    parameter_grids = {
        "Logistic Regression": {
            "classifier__C": [0.1, 1.0, 3.0, 10.0],
            "classifier__class_weight": [None, "balanced"]
        },

        "Decision Tree": {
            "classifier__max_depth": [8, 12, 18, None],
            "classifier__min_samples_split": [2, 10],
            "classifier__min_samples_leaf": [2, 5, 10],
            "classifier__class_weight": [None, "balanced"]
        },

        "KNN": {
            "classifier__n_neighbors": [5, 9, 15],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2]
        },

        "Naive Bayes": {
            "classifier__var_smoothing": [
                1e-11, 1e-10, 1e-9, 1e-8, 1e-7
            ]
        },

        "Support Vector Machine": {
            # A smaller search space keeps linear SVM tuning practical.
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__class_weight": [None, "balanced"]
        }
    }

    return models, parameter_grids


# ---------------------------------------------------------
# 5. Hyperparameter tuning
# ---------------------------------------------------------

def tune_models(models, parameter_grids, X_train, y_train):

    tuned_models = {}
    tuning_rows = []

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    tuning_results_path = ROOT / "tuning_results.csv"

    for name, model in models.items():

        print("\n" + "=" * 65)
        print(f"Hyperparameter tuning: {name}")
        print("=" * 65)

        try:
            if name == "Support Vector Machine":
                # Skip GridSearchCV for SVM: fit one efficient SGD-based linear SVM.
                # Hinge loss gives a linear SVM objective; SGD and early stopping
                # make training much faster on the Adult dataset.
                print("Training one SGD-based linear SVM (hinge loss; no GridSearchCV)...")
                model.fit(X_train, y_train)
                tuned_models[name] = model
                best_cv_f1 = "Not calculated (single fit)"
                best_parameters = {
                    "loss": "hinge",
                    "alpha": 1e-4,
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "tol": 1e-3,
                    "early_stopping": True
                }
            else:
                search = GridSearchCV(
                    estimator=model,
                    param_grid=parameter_grids[name],
                    scoring="f1",
                    cv=cv,
                    n_jobs=-1,
                    refit=True,
                    verbose=1,
                    error_score="raise"
                )
                search.fit(X_train, y_train)
                tuned_models[name] = search.best_estimator_
                best_cv_f1 = search.best_score_
                best_parameters = search.best_params_

            tuning_rows.append({
                "ML Model Name": name,
                "Best CV F1": best_cv_f1,
                "Best Parameters": str(best_parameters)
            })

            # Save results immediately after each model
            pd.DataFrame(tuning_rows).to_csv(
                tuning_results_path,
                index=False
            )

            print(f"\nCompleted: {name}")
            if isinstance(best_cv_f1, (int, float, np.floating)):
                print(f"Best CV F1: {best_cv_f1:.4f}")
            else:
                print(f"Best CV F1: {best_cv_f1}")
            print(f"Best Parameters: {best_parameters}")
            print(f"Tuning results saved: {tuning_results_path}")

        except Exception as e:

            print(f"\nERROR while tuning {name}: {str(e)}")
            raise

    return tuned_models

# ---------------------------------------------------------
# 6. Final evaluation on untouched test data
# ---------------------------------------------------------
def get_positive_scores(model, X):
    """Get continuous scores for ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_models(models, X_test, y_test):
    rows = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        predictions = model.predict(X_test)
        positive_scores = get_positive_scores(model, X_test)

        rows.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "AUC": roc_auc_score(y_test, positive_scores),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "MCC": matthews_corrcoef(y_test, predictions)
        })

        joblib.dump(
            model,
            MODEL_DIR / MODEL_FILE_NAMES[name]
        )
        print(f"Saved model: model/{MODEL_FILE_NAMES[name]}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 7. Generate comparison graphs
# ---------------------------------------------------------
def generate_graphs(metrics):
    metric_columns = [
        "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"
    ]

    plot_data = metrics.set_index("ML Model Name")[metric_columns]

    # Graph 1: all six metrics in one grouped bar chart.
    ax = plot_data.plot(
        kind="bar",
        figsize=(15, 8),
        width=0.82
    )

    ax.set_title(
        "Performance Comparison of Five Classification Models",
        fontsize=16,
        pad=18
    )
    ax.set_xlabel("Machine Learning Models", fontsize=12)
    ax.set_ylabel("Evaluation Score", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=15,
        ha="right"
    )
    ax.legend(
        title="Evaluation Metrics",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.45)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
            fontsize=7,
            padding=2
        )

    plt.tight_layout()
    plt.savefig(
        ROOT / "model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Graph 2: separate metric panels, easier to read and compare.
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for axis, metric in zip(axes, metric_columns):
        values = metrics.set_index("ML Model Name")[metric]
        bars = axis.bar(values.index, values.values)
        axis.set_title(metric, fontsize=12)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Score")
        axis.tick_params(axis="x", labelrotation=25)
        axis.grid(axis="y", linestyle="--", alpha=0.4)

        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")

        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    fig.suptitle(
        "Individual Evaluation Metric Comparison",
        fontsize=16
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(
        ROOT / "individual_metric_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nGenerated graphs:")
    print("- model_comparison.png")
    print("- individual_metric_comparison.png")


# ---------------------------------------------------------
# 8. Generate observations
# ---------------------------------------------------------
def make_observations(metrics):
    observations = {}

    descriptions = {
        "Logistic Regression":
            "A linear classification baseline. Its precision, recall and AUC "
            "show how effectively a linear decision boundary separates income classes.",

        "Decision Tree":
            "Captures non-linear relationships and feature interactions. "
            "Depth and leaf-size tuning helps control overfitting.",

        "KNN":
            "Uses distances to neighboring observations. Its results can be "
            "affected by feature scaling and the high-dimensional encoded feature space.",

        "Naive Bayes":
            "Provides a fast probabilistic baseline. Its conditional-independence "
            "assumption may be restrictive when census attributes are related.",

        "Support Vector Machine":
            "Uses an efficient linear maximum-margin classifier. Regularization tuning "
            "controls the trade-off between fitting the training data and generalization."
    }

    for _, row in metrics.iterrows():
        name = row["ML Model Name"]
        observations[name] = (
            f"Accuracy={row['Accuracy']:.4f}, AUC={row['AUC']:.4f}, "
            f"Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, "
            f"F1={row['F1']:.4f}, MCC={row['MCC']:.4f}. "
            + descriptions[name]
        )

    # F1 is the primary selection criterion; AUC and MCC break ties.
    winner = metrics.sort_values(
        ["F1", "AUC", "MCC"],
        ascending=False
    ).iloc[0]["ML Model Name"]

    observations["Overall, Winner for your dataset?"] = (
        f"{winner} is the winner under the selected F1-first ranking. "
        "The final choice should also consider AUC, MCC, accuracy and the "
        "relative cost of false positives and false negatives."
    )

    return observations


# ---------------------------------------------------------
# 9. Update README comparison and observations
# ---------------------------------------------------------
def update_readme(metrics, observations):
    readme_path = ROOT / "README.md"

    if not readme_path.exists():
        print("README.md not found. Skipping README update.")
        return

    content = readme_path.read_text(encoding="utf-8")
    start = "<!-- METRICS_START -->"
    end = "<!-- METRICS_END -->"

    if start not in content or end not in content:
        print("README metrics markers not found. Skipping README update.")
        return

    table_lines = [
        "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]

    for _, row in metrics.iterrows():
        table_lines.append(
            f"| {row['ML Model Name']} | {row['Accuracy']:.4f} | "
            f"{row['AUC']:.4f} | {row['Precision']:.4f} | "
            f"{row['Recall']:.4f} | {row['F1']:.4f} | "
            f"{row['MCC']:.4f} |"
        )

    observation_lines = [
        "| ML Model Name | Observation about model performance |",
        "|---|---|"
    ]

    for name in metrics["ML Model Name"]:
        safe_observation = observations[name].replace("|", "\\|")
        observation_lines.append(
            f"| {name} | {safe_observation} |"
        )

    winner_text = observations[
        "Overall, Winner for your dataset?"
    ].replace("|", "\\|")

    observation_lines.append(
        f"| Overall, Winner for your dataset? | {winner_text} |"
    )

    replacement = (
        start + "\n" +
        "\n".join(table_lines) + "\n\n" +
        "\n".join(observation_lines) + "\n" +
        end
    )

    before, remainder = content.split(start, 1)
    _, after = remainder.split(end, 1)

    readme_path.write_text(
        before + replacement + after,
        encoding="utf-8"
    )


# ---------------------------------------------------------
# 10. Generate PDF report with metrics and graph
# ---------------------------------------------------------
def create_pdf_report(metrics, observations, train_rows, test_rows, feature_count):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, Image, PageBreak
    )

    output = ROOT / "report.pdf"
    doc = SimpleDocTemplate(
        str(output),
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(
            "Adult Census Income Classification",
            styles["Title"]
        ),
        Spacer(1, 7),
        Paragraph("a. Problem statement", styles["Heading2"]),
        Paragraph(
            "The objective is to predict whether an individual's annual "
            "income is above $50,000 using census attributes. This is a "
            "binary classification task.",
            styles["BodyText"]
        ),
        Spacer(1, 5),
        Paragraph("b. Dataset description", styles["Heading2"]),
        Paragraph(
            f"The UCI Adult dataset contains {train_rows + test_rows:,} "
            f"records and 14 predictor features. The official adult.data "
            f"file ({train_rows:,} rows) is used for training and "
            f"cross-validation. The official adult.test file "
            f"({test_rows:,} rows) is reserved for final evaluation. "
            "Missing numerical values are median-imputed; missing "
            "categorical values are mode-imputed. Categorical features "
            "are one-hot encoded and numerical features are standardized.",
            styles["BodyText"]
        ),
        Spacer(1, 5),
        Paragraph("c. Github Repository Link", styles["Heading2"]),
        Paragraph(
            "Add your GitHub repository URL here after uploading the project.",
            styles["BodyText"]
        ),
        Spacer(1, 5),
        Paragraph("d. Models used and evaluation", styles["Heading2"]),
        Paragraph(
            "The five required classifiers are tuned using stratified "
            f"{CV_FOLDS}-fold cross-validation on the training partition. "
            "The selected models are evaluated once on the untouched "
            "official test partition. Hyperparameter selection uses F1.",
            styles["BodyText"]
        ),
        Spacer(1, 8)
    ]

    table_data = [[
        "Model", "Accuracy", "AUC", "Precision",
        "Recall", "F1", "MCC"
    ]]

    for _, row in metrics.iterrows():
        table_data.append([
            row["ML Model Name"],
            f"{row['Accuracy']:.4f}",
            f"{row['AUC']:.4f}",
            f"{row['Precision']:.4f}",
            f"{row['Recall']:.4f}",
            f"{row['F1']:.4f}",
            f"{row['MCC']:.4f}"
        ])

    table = Table(
        table_data,
        repeatRows=1,
        colWidths=[55*mm, 25*mm, 22*mm, 25*mm, 22*mm, 22*mm, 22*mm]
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5)
    ]))

    story.extend([
        table,
        Spacer(1, 8),
        Paragraph("Model performance observations", styles["Heading2"])
    ])

    for name in metrics["ML Model Name"]:
        story.append(Paragraph(
            f"<b>{html.escape(name)}:</b> "
            f"{html.escape(observations[name])}",
            styles["BodyText"]
        ))
        story.append(Spacer(1, 4))

    story.append(Paragraph(
        f"<b>Overall winner:</b> "
        f"{html.escape(observations['Overall, Winner for your dataset?'])}",
        styles["BodyText"]
    ))

    story.append(PageBreak())
    story.append(Paragraph(
        "Visual comparison of evaluation metrics",
        styles["Heading1"]
    ))

    graph_path = ROOT / "model_comparison.png"
    if graph_path.exists():
        story.append(Image(str(graph_path), width=250*mm, height=133*mm))

    doc.build(story)


# ---------------------------------------------------------
# 11. Main execution
# ---------------------------------------------------------
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    X_train, X_test, y_train, y_test = load_dataset()

    preprocessor = build_preprocessor(X_train)
    models, parameter_grids = build_models_and_parameters(preprocessor)


    # Hyperparameter tuning uses training data only.
    tuned_models = tune_models(
        models,
        parameter_grids,
        X_train,
        y_train
    )

    print("\nModel optimization completed.")

    # Final evaluation is performed on the untouched test set.
    metrics = evaluate_models(
        tuned_models,
        X_test,
        y_test
    )

    metrics.to_csv(ROOT / "metrics.csv", index=False)

    # Save held-out test data for the Streamlit application.
    test_data = X_test.copy()
    test_data["income"] = y_test.map({
        0: "<=50K",
        1: ">50K"
    })
    test_data.to_csv(ROOT / "test_data.csv", index=False)

    # Generate graphs from the measured metrics.
    generate_graphs(metrics)

    observations = make_observations(metrics)
    update_readme(metrics, observations)

    create_pdf_report(
        metrics,
        observations,
        len(X_train),
        len(X_test),
        X_train.shape[1]
    )

    print("\n" + "=" * 75)
    print("FINAL TEST-SET EVALUATION")
    print("=" * 75)
    print(metrics.round(4).to_string(index=False))

    winner = metrics.sort_values(
        ["F1", "AUC", "MCC"],
        ascending=False
    ).iloc[0]["ML Model Name"]

    print(f"\nOverall winner using F1-first ranking: {winner}")
    print("\nGenerated files:")
    print("- metrics.csv")
    print("- tuning_results.csv")
    print("- test_data.csv")
    print("- model/*.joblib")
    print("- model_comparison.png")
    print("- individual_metric_comparison.png")
    print("- report.pdf")
    print("- Updated README.md (if metrics markers exist)")


if __name__ == "__main__":
    main()