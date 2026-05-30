import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Create models folder automatically
os.makedirs("models", exist_ok=True)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/heart_disease_uci.csv")

print("\nFirst 5 Rows:\n")
print(df.head())

print("\nDataset Shape:\n")
print(df.shape)

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

# =========================
# HANDLE MISSING VALUES
# =========================

numeric_cols = df.select_dtypes(include='number').columns

df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].mean()
)

# =========================
# DATA VISUALIZATION
# =========================

# Histograms
df[numeric_cols].hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Feature Correlation Heatmap")
plt.show()

# =========================
# FEATURE & TARGET SPLIT
# =========================

X = df.drop("num", axis=1)

# Convert target into binary classification
# 0 = No Disease
# 1 = Disease

y = (df["num"] > 0).astype(int)

# =========================
# HANDLE CATEGORICAL DATA
# =========================

cat_cols = X.select_dtypes(include='object').columns

X = pd.get_dummies(X, columns=cat_cols)

print("\nFinal Feature Columns:\n")
print(X.columns)

# =========================
# TRAIN TEST SPLIT
# =========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FEATURE SCALING
# =========================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# =========================
# LOGISTIC REGRESSION MODEL
# =========================

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test_scaled)

# =========================
# RANDOM FOREST MODEL
# =========================

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)

# =========================
# MODEL EVALUATION
# =========================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Logistic Regression Accuracy
print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, y_pred_lr))

# Random Forest Accuracy
print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, y_pred_rf))

# Classification Report
print("\nRandom Forest Classification Report:\n")

print(classification_report(y_test, y_pred_rf))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix - Random Forest")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# =========================
# FEATURE IMPORTANCE
# =========================

feat_imp = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

plt.figure(figsize=(10, 6))

feat_imp.nlargest(10).plot(kind='barh')

plt.title("Top 10 Important Features")

plt.xlabel("Importance Score")

plt.show()

# =========================
# SAVE MODELS
# =========================

# Save Logistic Regression Model
joblib.dump(
    lr_model,
    "models/heart_model.pkl"
)

# Save Random Forest Model
joblib.dump(
    rf_model,
    "models/heart_rf_model.pkl"
)

# Save Scaler
joblib.dump(
    scaler,
    "models/scaler.pkl"
)

# Save Training Columns
joblib.dump(
    X.columns.tolist(),
    "models/training_columns.pkl"
)

print("\nModels and scaler saved successfully!")
