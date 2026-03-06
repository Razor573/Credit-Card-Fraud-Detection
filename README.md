# Credit Card Fraud Detection

A machine learning project exploring how to detect rare harmful events in
transaction data - with a focus on getting the evaluation framing right,
not just maximising accuracy.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Why I built this

Most fraud detection tutorials I found treat it as a standard classification
exercise - load data, train a model, report 99% accuracy, done.

That framing is wrong in an interesting way. A model that labels every single
transaction as "legitimate" would be 99.83% accurate on this dataset. Accuracy
is not just a bad metric here, it is actively misleading.

The real problem is: **how do you reliably detect rare harmful events without
creating so many false alarms that the system becomes useless?** That
tension between recall (catching fraud) and precision (not crying wolf) is
what I wanted to properly understand.

This also connects to something I find genuinely interesting: the same
problem structure - detecting rare, high-stakes events under distributional
skew - shows up in AI safety contexts too. A classifier that misses a
rare failure mode is dangerous in exactly the same way a fraud system that
misses rare frauds is costly.

---

## Dataset

**[Credit Card Fraud Detection - Kaggle (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**

- 284,807 transactions recorded over two days in September 2013
- 492 fraud cases - 0.173% of all transactions
- Features V1–V28: PCA-transformed (anonymised)
- `Amount` and `Time`: original, unscaled values

> Download `creditcard.csv` from Kaggle and place it in the project root.

---

## Project structure

```
Credit-Card-Fraud-Detection/
├── fraud_detection.py   # full pipeline: EDA → preprocessing → training → evaluation
├── requirements.txt
├── creditcard.csv       # not committed - download from Kaggle
├── figures/             # auto-generated plots (created on first run)
│   ├── 01_eda_overview.png
│   ├── 02_correlation_with_fraud.png
│   ├── 03_confusion_matrices.png
│   ├── 04_roc_pr_curves.png
│   ├── 05_feature_importance.png
│   └── 06_model_comparison.png
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Razor573/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Add dataset - download creditcard.csv from Kaggle, place in project root

python fraud_detection.py
```

---

## Methodology

### Exploratory analysis

Three questions I wanted to answer before touching a model:

1. How severe is the imbalance visually - does it look as bad as 0.17% sounds?
2. Do fraud transactions differ from legitimate ones in amount or timing?
3. Which PCA features correlate most with the fraud label?

The amount distribution was the most surprising: fraud transactions tend to be
*smaller* on average, not larger. A naive rule like "flag high-value
transactions" would miss most of them.

### Preprocessing

`Amount` and `Time` are scaled with `StandardScaler`. V1–V28 are already
PCA-transformed by the dataset authors and need no further scaling.

Stratified 80/20 train-test split - without stratification, a random split
could put a disproportionate number of fraud cases on one side.

### Handling class imbalance - SMOTE

SMOTE (Synthetic Minority Oversampling Technique) generates synthetic fraud
samples during training to rebalance the classes.

The important detail: SMOTE is placed **inside** an `imblearn.Pipeline` so it
only sees training data. Applying it before splitting would cause synthetic
samples to leak into the test set, inflating recall scores artificially.

```python
pipe = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf",   RandomForestClassifier(...))
])
```

### Models compared

Three models with increasing complexity:

| Model | Key setting |
|---|---|
| Logistic Regression | `class_weight="balanced"` + SMOTE |
| Random Forest | 100 trees; gives feature importances |
| XGBoost | `scale_pos_weight=577` ≈ negative/positive ratio |

### Evaluation metrics

**Why not accuracy?** Already covered above - it is useless here.

Primary metric: **PR-AUC (Average Precision)**. It measures model quality
across every possible decision threshold on the minority class. A model
that happens to be good at the easy 99.83% majority gets no credit.

Secondary: **ROC-AUC**, **recall**, **precision**, and confusion matrix.

---

## Results

| Model | ROC-AUC | PR-AUC | Fraud Recall |
|---|---|---|---|
| Logistic Regression | ~0.978 | ~0.720 | ~0.91 |
| Random Forest | ~0.985 | ~0.854 | ~0.82 |
| XGBoost | ~0.983 | ~0.868 | ~0.84 |

Random Forest and XGBoost are close on PR-AUC. Logistic Regression has
higher recall but lower precision - it catches more fraud but raises more
false alarms. Which you prefer depends on the cost trade-off.

---

## Business impact simulation

Rather than reporting F1 scores in isolation, the script estimates the
financial impact of deploying the best model on the test set:

```
── Business impact simulation (XGBoost) ──
  Fraud caught   (TP):    82   → ~$10,021 recovered
  Fraud missed   (FN):    16   → ~$1,955 lost
  False alerts   (FP):    18   (legitimate transactions flagged)
  Correct clears (TN): 56848

  Recovery rate: 83.7%
  Analyst workload: 100 flagged transactions per 56,962 processed
```

This is how fraud teams actually think about model value.

---

## Limitations and what I would do next

- **No SHAP values** - the PCA features are opaque; in a real deployment
  you would want SHAP to explain individual predictions to fraud analysts
- **Static dataset** - fraud patterns drift; a deployed model would need
  periodic retraining or drift monitoring
- **Fixed threshold** - the default 0.5 threshold is arbitrary; optimal
  threshold depends on the business cost of a missed fraud vs a false alert
- Possible extensions: cost-sensitive learning, isolation forest as an
  unsupervised complement, a lightweight FastAPI endpoint for real-time scoring

---

## Dependencies

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
xgboost>=2.0
```

---

## License

MIT - use freely, adapt, build on.

---

**Vagif Novruzov** - [LinkedIn](https://linkedin.com/in/vagif-novruzov) · [GitHub](https://github.com/Razor573)
