# Card-Not-Present Fraud Detection using SVM

A Support Vector Machine (SVM) based classifier for detecting fraudulent, card-not-present
e-commerce transactions. Built to explore kernel choice, hyperparameter tuning, and feature
selection on a highly imbalanced, rare-event classification problem — a common real-world
challenge in FinTech risk systems.

## Problem

Online payment gateways process thousands of transactions per minute, and fraud events are rare
relative to legitimate activity but expensive when missed (chargebacks, merchant trust loss).
This project builds and evaluates an SVM classifier suited to a high-dimensional, non-linearly
separable feature space.

## Dataset

- **Name:** Credit Card Fraud Detection
- **Source:** Kaggle — `mlg-ulb/creditcardfraud`
- **URL:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions, 31 features (Time, Amount, V1–V28 PCA-anonymized), ~0.17% fraud rate

## What's Covered

- Exploratory Data Analysis — class imbalance quantification, feature separability checks
- Feature scaling and stratified train/test splitting
- Baseline Linear SVM vs. non-linear RBF and Polynomial kernels
- Stratified K-Fold cross-validation
- Hyperparameter tuning (`C`, `gamma`) via GridSearchCV
- Correlation-based feature selection with before/after comparison
- Evaluation using precision, recall, F1-score, and PR-AUC (chosen over plain accuracy due to
  severe class imbalance)
- Discussion of the operational cost tradeoff: false positives (blocked legitimate customers)
  vs. false negatives (missed fraud)

## Tech Stack

Python · scikit-learn · pandas · NumPy · matplotlib · seaborn

## Results
|index|Configuration|Precision|Recall|F1-Score|PR-AUC|Training Time \(s\)|
|---|---|---|---|---|---|---|
|0|Linear SVM \(baseline\)|0\.0|0\.0|0\.0|0\.5|0\.011396408081054688|
|1|RBF SVM|0\.9333333333333333|0\.875|0\.9032258064516129|0\.9162583487419014|0\.24170303344726562|
|2|Polynomial SVM \(degree=3\)|0\.7894736842105263|0\.9375|0\.8571428571428571|0\.9740239845938375|0\.13407540321350098|
|3|Tuned RBF SVM \(GridSearchCV\)|0\.8333333333333334|0\.9375|0\.8823529411764706|0\.805078438181186|138\.46695184707642|
|4|Tuned RBF SVM \(Reduced Features\)|0\.8333333333333334|0\.9375|0\.8823529411764706|0\.805078438181186|0\.08736228942871094|
## How to Run

1. Clone this repo
2. Download `creditcard.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
3. Open `SVM_Fraud_Detection.ipynb` in Google Colab or Jupyter
4. Upload `creditcard.csv` when prompted
5. Run all cells top-to-bottom

## Author

Amrita Jadhav