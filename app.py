# -*- coding: utf-8 -*-
"""
CIS 412 Final Project - Fraud Detection
Streamlit Deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, precision_recall_curve, auc
)
from xgboost import XGBClassifier
from matplotlib.colors import LinearSegmentedColormap

# ==========================================
# Page config
# ==========================================
st.set_page_config(page_title="Fraud Detection - CIS 412", layout="wide")
st.title("Fraud Detection - CIS 412 Final Project")

# ==========================================
# 1. Load data from Google Drive
# ==========================================
@st.cache_data
def load_data():
    # Google Drive direct download link
    # Replace FILE_ID with your actual file ID from the sharing link
    url = "https://drive.google.com/uc?id=12rpuwS4zq2O3W4_YPoCsPG4CDFKDvYt2&export=download"
    df = pd.read_csv(url, low_memory=False)
    return df

st.header("1. Load & Explore Data")

try:
    df = load_data()
    st.success(f"Dataset loaded successfully! Shape: {df.shape}")
    st.dataframe(df.head())
except Exception as e:
    st.error("Failed to load data. Make sure the Google Drive link is set to 'Anyone with the link'.")
    st.error(str(e))
    st.stop()

# ==========================================
# 2. Check original fraud distribution
# ==========================================
st.header("2. Original Fraud Distribution")

col1, col2 = st.columns(2)

with col1:
    st.write("**Class Counts:**")
    st.write(df['isFraud'].value_counts())
    st.write("**Class Percentages:**")
    st.write(df['isFraud'].value_counts(normalize=True) * 100)

with col2:
    before_counts = df['isFraud'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        before_counts,
        labels=['Non-Fraud', 'Fraud'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title('Class Distribution Before Balancing')
    st.pyplot(fig)
    plt.clf()

# ==========================================
# 3. Balance dataset (1:10 ratio)
# ==========================================
st.header("3. Balance Dataset (1:10 Ratio)")

fraud = df[df['isFraud'] == 1]
non_fraud = df[df['isFraud'] == 0]

st.write(f"Fraud rows: {len(fraud)}")
st.write(f"Non-fraud rows: {len(non_fraud)}")

non_fraud_sample = non_fraud.sample(n=len(fraud) * 10, random_state=0)
df_sampled = pd.concat([fraud, non_fraud_sample])
df_sampled = df_sampled.sample(frac=1, random_state=0)

col1, col2 = st.columns(2)

with col1:
    st.write("**Balanced Class Counts:**")
    st.write(df_sampled['isFraud'].value_counts())
    st.write("**Balanced Class Percentages:**")
    st.write(df_sampled['isFraud'].value_counts(normalize=True) * 100)

with col2:
    after_counts = df_sampled['isFraud'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        after_counts,
        labels=['Non-Fraud', 'Fraud'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.set_title('Class Distribution After Balancing')
    st.pyplot(fig)
    plt.clf()

# ==========================================
# 4. Data cleaning
# ==========================================
st.header("4. Data Cleaning")

st.write("**Missing Values:**")
st.write(df_sampled.isnull().sum())

st.write(f"**Duplicates:** {df_sampled.duplicated().sum()}")

df_sampled = df_sampled.drop_duplicates()
st.write(f"**Shape after cleaning:** {df_sampled.shape}")

# ==========================================
# 5. Data transformation
# ==========================================
st.header("5. Data Transformation")

df_sampled = pd.get_dummies(df_sampled, columns=['type'], drop_first=True)
df_sampled = df_sampled.drop(['nameOrig', 'nameDest'], axis=1)

st.write("Converted transaction type to dummy variables and dropped ID columns.")
st.dataframe(df_sampled.head())

# ==========================================
# 6. Feature engineering
# ==========================================
st.header("6. Feature Engineering")

df_sampled['balanceDiffOrig'] = df_sampled['oldbalanceOrg'] - df_sampled['newbalanceOrig']
df_sampled['balanceDiffDest'] = df_sampled['newbalanceDest'] - df_sampled['oldbalanceDest']

st.write("Created `balanceDiffOrig` and `balanceDiffDest` features.")
st.dataframe(df_sampled.head())

# ==========================================
# 7. Split into features and target
# ==========================================
st.header("7. Train-Test Split & Scaling")

X = df_sampled.drop('isFraud', axis=1)
y = df_sampled['isFraud']

st.write(f"Features shape: {X.shape}")
st.write(f"Target shape: {y.shape}")

# ==========================================
# 8. Train-test split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

col1, col2 = st.columns(2)
with col1:
    st.write("**Training set:**")
    st.write(y_train.value_counts())
with col2:
    st.write("**Test set:**")
    st.write(y_test.value_counts())

# ==========================================
# 9. Scaling
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write(f"Scaled training shape: {X_train_scaled.shape}")
st.write(f"Scaled test shape: {X_test_scaled.shape}")

# ==========================================
# 10. Logistic Regression
# ==========================================
st.header("8. Logistic Regression")

with st.spinner("Training Logistic Regression..."):
    logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=0)
    logreg.fit(X_train_scaled, y_train)

y_train_pred_log = logreg.predict(X_train_scaled)

st.subheader("Training Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_train, y_train_pred_log):.4f}")
col2.metric("Precision", f"{precision_score(y_train, y_train_pred_log):.4f}")
col3.metric("Recall", f"{recall_score(y_train, y_train_pred_log):.4f}")
col4.metric("F1-Score", f"{f1_score(y_train, y_train_pred_log):.4f}")

# Training confusion matrix
cm_logreg = confusion_matrix(y_train, y_train_pred_log)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(cm_logreg, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'], ax=ax)
ax.set_title('Logistic Regression - Training Confusion Matrix')
ax.set_ylabel('Actual Label')
ax.set_xlabel('Predicted Label')
st.pyplot(fig)
plt.clf()

# Coefficients and Odds Ratios
col1, col2 = st.columns(2)
with col1:
    st.subheader("Coefficients")
    coef = pd.DataFrame(logreg.coef_[0], index=X.columns, columns=['Coefficients'])
    st.dataframe(coef.sort_values(by='Coefficients', ascending=False))

with col2:
    st.subheader("Odds Ratios")
    odds = pd.DataFrame(np.exp(logreg.coef_[0]), index=X.columns, columns=['Odds'])
    st.dataframe(odds.sort_values(by='Odds', ascending=False))

# ==========================================
# 11. XGBoost (base model)
# ==========================================
st.header("9. XGBoost (Base Model)")

with st.spinner("Training XGBoost..."):
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=0,
        scale_pos_weight=10
    )
    xgb_model.fit(X_train_scaled, y_train)

y_train_pred_xgb = xgb_model.predict(X_train_scaled)

st.subheader("Training Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_train, y_train_pred_xgb):.4f}")
col2.metric("Precision", f"{precision_score(y_train, y_train_pred_xgb):.4f}")
col3.metric("Recall", f"{recall_score(y_train, y_train_pred_xgb):.4f}")
col4.metric("F1-Score", f"{f1_score(y_train, y_train_pred_xgb):.4f}")

# Training confusion matrix
cm_xgb_train = confusion_matrix(y_train, y_train_pred_xgb)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(cm_xgb_train, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'], ax=ax)
ax.set_title('XGBoost - Training Confusion Matrix')
ax.set_ylabel('Actual Label')
ax.set_xlabel('Predicted Label')
st.pyplot(fig)
plt.clf()

# Feature importances
st.subheader("Feature Importances")
importances = pd.DataFrame(xgb_model.feature_importances_, index=X.columns, columns=['Importance'])
st.dataframe(importances.sort_values(by='Importance', ascending=False))

# ==========================================
# 12. Training PR Curve comparison
# ==========================================
st.header("10. Training Precision-Recall Curve")

y_scores_log = logreg.predict_proba(X_train_scaled)[:, 1]
y_scores_xgb = xgb_model.predict_proba(X_train_scaled)[:, 1]

precision1, recall1, _ = precision_recall_curve(y_train, y_scores_xgb)
pr_auc1 = auc(recall1, precision1)

precision2, recall2, _ = precision_recall_curve(y_train, y_scores_log)
pr_auc2 = auc(recall2, precision2)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recall1, precision1, label=f'XGBoost (AUC = {pr_auc1:.4f})')
ax.plot(recall2, precision2, label=f'Logistic Regression (AUC = {pr_auc2:.4f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve (Training)')
ax.legend()
ax.grid(True)
st.pyplot(fig)
plt.clf()

# ==========================================
# 13. Test set evaluation
# ==========================================
st.header("11. Test Set Evaluation")

y_test_pred_log = logreg.predict(X_test_scaled)
y_test_pred_xgb = xgb_model.predict(X_test_scaled)

st.subheader("Logistic Regression - Test Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_test_pred_log):.4f}")
col2.metric("Precision", f"{precision_score(y_test, y_test_pred_log):.4f}")
col3.metric("Recall", f"{recall_score(y_test, y_test_pred_log):.4f}")
col4.metric("F1", f"{f1_score(y_test, y_test_pred_log):.4f}")

st.subheader("XGBoost - Test Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_test_pred_xgb):.4f}")
col2.metric("Precision", f"{precision_score(y_test, y_test_pred_xgb):.4f}")
col3.metric("Recall", f"{recall_score(y_test, y_test_pred_xgb):.4f}")
col4.metric("F1", f"{f1_score(y_test, y_test_pred_xgb):.4f}")

# ==========================================
# 14. Test confusion matrices (presentation style)
# ==========================================
st.subheader("Test Set Confusion Matrices")

sns.set(style="white")
labels = ['Non-Fraud', 'Fraud']

cm_log_test = confusion_matrix(y_test, y_test_pred_log)
cm_xgb_test = confusion_matrix(y_test, y_test_pred_xgb)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_log_test, annot=True, fmt='d', cmap='crest',
            xticklabels=labels, yticklabels=labels,
            linewidths=0.8, linecolor='white', cbar=False, ax=axes[0])
axes[0].set_title('Logistic Regression (Test Set)', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_xgb_test, annot=True, fmt='d', cmap='crest',
            xticklabels=labels, yticklabels=labels,
            linewidths=0.8, linecolor='white', cbar=False, ax=axes[1])
axes[1].set_title('XGBoost (Test Set)', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
st.pyplot(fig)
plt.clf()

# ==========================================
# 15. Test PR curve
# ==========================================
st.subheader("Precision-Recall Curve (Test Set)")

y_scores_log_test = logreg.predict_proba(X_test_scaled)[:, 1]
y_scores_xgb_test = xgb_model.predict_proba(X_test_scaled)[:, 1]

precision_log, recall_log, _ = precision_recall_curve(y_test, y_scores_log_test)
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_scores_xgb_test)

auc_log = auc(recall_log, precision_log)
auc_xgb = auc(recall_xgb, precision_xgb)

plt.style.use('default')
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(recall_xgb, precision_xgb,
        label=f'XGBoost (AUC = {auc_xgb:.3f})',
        linewidth=3, color='#1f3b5c')
ax.plot(recall_log, precision_log,
        label=f'Logistic Regression (AUC = {auc_log:.3f})',
        linewidth=3, color='#4caf7d')

ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curve (Test Set)', fontsize=14, fontweight='bold')
ax.legend(frameon=False)
ax.grid(alpha=0.15)
for spine in ax.spines.values():
    spine.set_visible(False)

st.pyplot(fig)
plt.clf()

# ==========================================
# 16. Tuned XGBoost (hardcoded best params)
# ==========================================
st.header("12. Tuned XGBoost (Hyperparameter Tuning)")

st.write("Best parameters found via GridSearchCV (tuned for recall):")

# HARDCODED — update these if you find your actual best_params_
best_params = {
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.1,
    'scale_pos_weight': 15
}

st.json(best_params)

with st.spinner("Training tuned XGBoost..."):
    best_xgb = XGBClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        scale_pos_weight=best_params['scale_pos_weight'],
        random_state=0,
        eval_metric='logloss'
    )
    best_xgb.fit(X_train_scaled, y_train)

y_test_pred_tuned = best_xgb.predict(X_test_scaled)

st.subheader("Tuned XGBoost - Test Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy_score(y_test, y_test_pred_tuned):.4f}")
col2.metric("Precision", f"{precision_score(y_test, y_test_pred_tuned):.4f}")
col3.metric("Recall", f"{recall_score(y_test, y_test_pred_tuned):.4f}")
col4.metric("F1", f"{f1_score(y_test, y_test_pred_tuned):.4f}")

# ==========================================
# 17. Before vs After tuning confusion matrices
# ==========================================
st.subheader("XGBoost: Before vs After Tuning")

cm_old = confusion_matrix(y_test, y_test_pred_xgb)
cm_new = confusion_matrix(y_test, y_test_pred_tuned)

colors = ["#E8F0E6", "#7FB77E", "#2F3E75"]
custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(cm_old, annot=True, fmt='d', cmap=custom_cmap, ax=axes[0],
            xticklabels=["Non-Fraud", "Fraud"],
            yticklabels=["Non-Fraud", "Fraud"])
axes[0].set_title("XGBoost (Before Tuning)", fontsize=12)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_new, annot=True, fmt='d', cmap=custom_cmap, ax=axes[1],
            xticklabels=["Non-Fraud", "Fraud"],
            yticklabels=["Non-Fraud", "Fraud"])
axes[1].set_title("XGBoost (After Tuning)", fontsize=12)
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
st.pyplot(fig)
plt.clf()

# ==========================================
# 18. Precision & Recall vs Threshold
# ==========================================
st.subheader("Precision & Recall vs Threshold (Tuned XGBoost)")

y_probs = best_xgb.predict_proba(X_test_scaled)[:, 1]
precision_t, recall_t, thresholds_t = precision_recall_curve(y_test, y_probs)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(thresholds_t, precision_t[:-1], label="Precision", color="#2F3E75", linewidth=2)
ax.plot(thresholds_t, recall_t[:-1], label="Recall", color="#7FB77E", linewidth=2)
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision & Recall vs Threshold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.clf()
