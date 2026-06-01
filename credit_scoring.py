import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
accuracy_score,
classification_report,
roc_auc_score,
confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load Dataset

df = pd.read_excel("dataset/credit_risk_dataset.xlsx")

# Display first 5 rows

print(df.head())

# Check missing values

print(df.isnull().sum())

# Remove missing values

df = df.dropna()

# Encode categorical columns

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
df[col] = le.fit_transform(df[col])

# Features and Target

X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

# Train Model

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# Prediction

pred = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

# Classification Report

print("\nClassification Report:")
print(classification_report(y_test, pred))

# ROC AUC Score

prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, prob)

print("ROC AUC Score:", roc_auc)

# Confusion Matrix

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save Model

joblib.dump(model, "credit_scoring_model.pkl")

print("Model saved successfully!")
