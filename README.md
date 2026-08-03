# Credit Card Fraud Detection

This repository contains a complete example of a credit card fraud detection project using Apache Spark ML.
It includes a Jupyter notebook, a runnable Python script, and a real fraud dataset.

## Overview

The goal of this repository is to demonstrate how to build, train, and evaluate a supervised machine learning pipeline
for fraud detection using transaction-level data.

The project uses a public credit card fraud dataset with anonymized features and a binary target column named `Class`.
The model is trained using a `RandomForestClassifier` in Spark ML and evaluated using standard classification metrics.

## Contents

- `Spark ML - Credit card Fraud.ipynb`
  - A notebook with exploratory data analysis, Spark DataFrame operations, feature assembly, model training, and evaluation.
- `run_credit_card_fraud.py`
  - A standalone Python script that loads the dataset, builds a Spark pipeline, trains a model, and prints evaluation results.
- `data/creditcard-fraud.csv`
  - The dataset used by the notebook and the Python script.
- `README.md`
  - This file.

## Dataset details

The dataset is a credit card fraud dataset with the following characteristics:

- Features: `V1` through `V28`, plus `Amount`
- Label: `Class`
- Label values:
  - `0` = non-fraudulent transaction
  - `1` = fraudulent transaction
- The dataset is balanced for the purposes of this example.

The dataset is stored locally in the repository under `data/creditcard-fraud.csv`.

## Project structure

This repository is organized so the notebook and script can be executed independently.

### `Spark ML - Credit card Fraud.ipynb`

This notebook contains:

1. Data loading with Spark DataFrame reader
2. Schema inspection and sample data display
3. Feature selection and vector assembly
4. Train/test split
5. Model training with Spark ML pipeline
6. Model evaluation using AUC, accuracy, and F1

### `run_credit_card_fraud.py`

This Python script is the main executable component for the project.
It performs the following steps:

1. Loads the dataset from `data/creditcard-fraud.csv`
2. Selects feature columns `V1..V28` and `Amount`
3. Builds a `VectorAssembler` for feature transformation
4. Uses `RandomForestClassifier` with `Class` as the label column
5. Splits the data into training and test sets
6. Trains the model using a Spark pipeline
7. Evaluates the model and prints metrics

## Setup instructions

To run the project locally, follow these steps.

### 1. Install Python dependencies

Install the required Python packages using pip:

```bash
python3 -m pip install pyspark numpy
```

This installs:

- `pyspark` — Apache Spark support for Python
- `numpy` — numerical library required by Spark ML

### 2. Verify the dataset

Confirm that the dataset exists in the repository:

```bash
ls -lh data/creditcard-fraud.csv
```

If the file is missing, restore it from the repository or contact the project owner.

## Running the project script

Run the Python script from the repository root:

```bash
python3 run_credit_card_fraud.py
```

The script will:

- start a Spark session
- load the dataset
- print the DataFrame schema
- display sample rows
- train a RandomForest model
- print AUC, accuracy, and F1 results

## Expected output

The script should print the loaded schema and sample data rows.
It should also display evaluation metrics similar to:

- `AUC`: a numeric score between 0.0 and 1.0
- `Accuracy (MulticlassEvaluator)`: a numeric score between 0.0 and 1.0
- `F1 score`: a numeric score between 0.0 and 1.0
- `Accuracy (direct compare)`: a numeric score between 0.0 and 1.0

A successful run means the Spark pipeline executed without errors.

## Notes on dataset and evaluation

The dataset uses anonymized transaction features, which is common practice for fraud datasets.
`V1..V28` are transformed or anonymized feature values.
The `Amount` column represents the transaction amount.

The model uses a random forest because it handles numerical features well and is robust to noisy data.

### Why AUC is useful

AUC (Area Under the ROC Curve) is a useful metric for binary classification because it measures how well
models distinguish between positive and negative classes.

When fraud detection datasets are imbalanced or class distributions vary, AUC can give a better sense
of performance than accuracy alone.

### Why accuracy can be misleading

Accuracy is useful, but if classes are imbalanced, a high accuracy score may simply reflect the dominant class.
In fraud detection, a model that predicts every transaction as non-fraud may achieve high accuracy but poor fraud detection.

### F1 score

F1 score balances precision and recall, which is helpful for fraud detection where false negatives and false positives both matter.

## Troubleshooting

### Common issues

- `ModuleNotFoundError: No module named 'pyspark'`
  - Install `pyspark` with `python3 -m pip install pyspark`.

- `No such file or directory: data/creditcard-fraud.csv`
  - Verify the file exists at `data/creditcard-fraud.csv`.
  - If needed, restore the dataset file from the repository.

- Spark warnings about native Hadoop libraries
  - These warnings are normal in many environments and do not usually affect execution.

- `ValueError` or schema mismatch
  - Confirm the dataset has the expected feature names: `V1..V28`, `Amount`, and `Class`.

## Development notes

If you want to extend this project, consider these improvements:

- Add feature scaling or normalization
- Add a second model such as `LogisticRegression` or `GBTClassifier`
- Add cross-validation or parameter tuning
- Add a confusion matrix and classification report
- Save the trained model to disk for future use
- Add a separate notebook for feature engineering

## File descriptions

### `Spark ML - Credit card Fraud.ipynb`

This notebook includes the original exploratory and training workflow.
It is useful for interactive analysis, data exploration, and incremental experiment tracking.

### `run_credit_card_fraud.py`

This script is useful for automated execution, testing, and running the project in non-interactive environments.
It is the recommended entry point if you want to reproduce the model training and evaluation.

### `data/creditcard-fraud.csv`

This is the dataset file used by both the notebook and the script.
It is stored in the `data` directory so the project can reference it with a local relative path.

## Repository usage guidelines

To reproduce results, run the script from the repository root.
Do not move the dataset file unless you also update the file path in `run_credit_card_fraud.py` and the notebook.

If you want to add additional datasets, create a new file under `data/` and update the loading path.

## License and attribution

This repository is intended for educational and demonstration purposes.
The dataset is a publicly available credit card fraud dataset used for machine learning examples.

## Contact

If you need help with the project, use the repository issue tracker or contact the maintainer.

---

Thank you for using this credit card fraud detection example repository.
