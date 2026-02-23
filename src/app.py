

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="KNN Diabetes Classifier", layout="centered")
st.title("🧠 KNN Classification – Pima Diabetes Dataset")

df = pd.read_csv("data/pima_indians_diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

k = st.slider("Choose K (neighbors)", min_value=1, max_value=25, value=5, step=2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

st.subheader("✅ Model Performance")
st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))

st.subheader("🔮 Predict with Custom Inputs")
inputs = []
for col in X.columns:
    val = st.number_input(col, value=float(X[col].median()))
    inputs.append(val)

if st.button("Predict"):
    scaled = scaler.transform([inputs])
    pred = model.predict(scaled)[0]
    st.success(f"Prediction (0=No Diabetes, 1=Diabetes): **{pred}**")