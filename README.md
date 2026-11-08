# Adult Census Income Classification

## a. Problem statement

The objective of this project is to build and compare five supervised machine-learning
classification models to predict whether an individual's annual income exceeds $50,000
based on census-related attributes. This is a binary classification problem with two
classes: `<=50K` and `>50K`.

## b. Dataset description

- **Dataset:** Adult (Census Income)
- **Repository:** UCI Machine Learning Repository
- **Dataset URL:** https://archive.ics.uci.edu/dataset/2/adult
- **Instances:** 48,842
- **Input features:** 14
- **Target:** Income class (`<=50K` or `>50K`)
- **Feature types:** Categorical and integer/numerical
- **Missing values:** Present in attributes such as workclass, occupation, and native-country.

The dataset was extracted from the 1994 US Census database. The classification task is
to determine whether a person's annual income is greater than $50,000. Missing values
are imputed inside the model pipelines. Categorical variables are one-hot encoded and
numerical variables are median-imputed and standardized.

## c. Github Repository Link

**GitHub URL:** https://github.com/Varsha180/adult-income-classification.git

## d. Models used

The following five models are trained and evaluated on the same dataset using an
80:20 stratified train-test split and a fixed random seed (`42`):

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine (linear kernel)

The evaluation metrics are Accuracy, AUC, Precision, Recall, F1 Score, and Matthews
Correlation Coefficient (MCC). AUC is calculated from predicted probabilities where
available and from decision scores for the SVM.

### Comparison table

Run `python train_models.py`. The script automatically replaces the table below with
the measured results from the held-out test data.

<!-- METRICS_START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8057 | 0.9032 | 0.5591 | 0.8398 | 0.6713 | 0.5637 |
| Decision Tree | 0.8078 | 0.8879 | 0.5616 | 0.8495 | 0.6762 | 0.5712 |
| KNN | 0.8437 | 0.8889 | 0.6935 | 0.6061 | 0.6469 | 0.5491 |
| Naive Bayes | 0.5880 | 0.8090 | 0.3568 | 0.9272 | 0.5153 | 0.3579 |
| Support Vector Machine | 0.7914 | 0.8948 | 0.5370 | 0.8495 | 0.6580 | 0.5469 |

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy=0.8057, AUC=0.9032, Precision=0.5591, Recall=0.8398, F1=0.6713, MCC=0.5637. A linear classification baseline. Its precision, recall and AUC show how effectively a linear decision boundary separates income classes. |
| Decision Tree | Accuracy=0.8078, AUC=0.8879, Precision=0.5616, Recall=0.8495, F1=0.6762, MCC=0.5712. Captures non-linear relationships and feature interactions. Depth and leaf-size tuning helps control overfitting. |
| KNN | Accuracy=0.8437, AUC=0.8889, Precision=0.6935, Recall=0.6061, F1=0.6469, MCC=0.5491. Uses distances to neighboring observations. Its results can be affected by feature scaling and the high-dimensional encoded feature space. |
| Naive Bayes | Accuracy=0.5880, AUC=0.8090, Precision=0.3568, Recall=0.9272, F1=0.5153, MCC=0.3579. Provides a fast probabilistic baseline. Its conditional-independence assumption may be restrictive when census attributes are related. |
| Support Vector Machine | Accuracy=0.7914, AUC=0.8948, Precision=0.5370, Recall=0.8495, F1=0.6580, MCC=0.5469. Uses an efficient linear maximum-margin classifier. Regularization tuning controls the trade-off between fitting the training data and generalization. |
| Overall, Winner for your dataset? | Decision Tree is the winner under the selected F1-first ranking. The final choice should also consider AUC, MCC, accuracy and the relative cost of false positives and false negatives. |
<!-- METRICS_END -->

### Observations and winner

The script generates model-specific observations from the measured evaluation metrics.
The overall winner is selected using F1 score, with AUC and MCC used as tie-breakers.
This is a metric-based selection; the best model for a real application also depends
on the relative cost of false-positive and false-negative predictions.