# Credit Card Fraud Detection

This repository contains a Spark ML project for detecting credit card fraud.

## Files
- `Spark ML - Credit card Fraud.ipynb` — notebook with Spark ML code
- `run_credit_card_fraud.py` — runnable Python script for training and evaluating a RandomForest model
- `data/creditcard-fraud.csv` — fraud dataset used by the notebook and script

## Run the project
1. Install dependencies:
   - `python3 -m pip install pyspark numpy`
2. Run the script:
   - `python3 run_credit_card_fraud.py`

The script loads `data/creditcard-fraud.csv`, trains a model on features `V1..V28` and `Amount`, and evaluates using AUC, accuracy, and F1.
