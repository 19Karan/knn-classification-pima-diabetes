# KNN Classification – Pima Indians Diabetes

## Overview
This project applies **K-Nearest Neighbors (KNN)** to predict diabetes outcomes using the Pima Indians Diabetes dataset.  
It includes preprocessing (scaling), k-value tuning, and evaluation.

## Dataset
- `data/pima_indians_diabetes.csv`

## Key Steps
- Train-test split
- Feature scaling (StandardScaler)
- K selection (accuracy vs k)
- Evaluation: accuracy + confusion matrix

## Project Structure

knn-classification-pima-diabetes/
├── data/ Pima Dataset
├── notebooks/ Knn.ipynb
├── src/ app.py
├── README.md
├── requirements.txt
└── .gitignore


## Run
```bash
pip install -r requirements.txt
streamlit run src/app.py
