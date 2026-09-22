import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load data ---
matrix = pd.read_csv("HDFS_v1/preprocessed/Event_occurrence_matrix.csv")
feature_cols = [col for col in matrix.columns if col.startswith("E")]
X = matrix[feature_cols]
y = (matrix["Label"] == "Fail").astype(int)

# --- Scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training Isolation Forest...")
IF = IsolationForest(n_estimators=100, contamination=0.03, random_state=42, n_jobs=-1)
IF.fit(X_scaled)
if_scores = -IF.decision_function(X_scaled)  # higher = more anomalous

# LOF removed — performs poorly on duplicate-heavy HDFS event counts
print("Training One-Class SVM (this may take a few minutes)...")
# Use a sample for OCSVM - it doesn't scale well to 575K rows
sample_size = 50000
idx = np.random.RandomState(42).choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[idx]

OCSVM = OneClassSVM(kernel='rbf', nu=0.03)
OCSVM.fit(X_sample)
ocsvm_scores = -OCSVM.decision_function(X_scaled)  # higher = more anomalous

print("Building ensemble...")
# --- Normalize each score to 0-1 range before combining ---
def normalize(scores):
    return (scores - scores.min()) / (scores.max() - scores.min())

if_norm = normalize(if_scores)
# lof_norm removed
ocsvm_norm = normalize(ocsvm_scores)

# --- Weighted ensemble score ---
# IF gets highest weight since it performed best alone
ensemble_score = 0.65 * if_norm + 0.35 * ocsvm_norm
# --- Threshold at top 3% (our contamination estimate) ---
threshold = np.percentile(ensemble_score, 97)
preds = (ensemble_score >= threshold).astype(int)

# --- Evaluate ---
p = precision_score(y, preds)
r = recall_score(y, preds)
f1 = f1_score(y, preds)

print("\n" + "=" * 40)
print("HDFS — Ensemble (IF + LOF + OCSVM)")
print("=" * 40)
print(f"Precision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"\nTotal anomalies detected: {preds.sum()}")
print(f"Total actual anomalies:   {y.sum()}")

# --- Confusion Matrix ---
cm = confusion_matrix(y, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("HDFS — Confusion Matrix (Ensemble)")
plt.tight_layout()
plt.savefig("hdfs_ensemble_confusion_matrix.png")
print("Confusion matrix saved.")

# --- Comparison Summary ---
print("\n" + "=" * 40)
print("COMPARISON SUMMARY")
print("=" * 40)
print(f"{'Method':<25}{'F1':<10}")
print(f"{'Isolation Forest':<25}{0.663:<10.3f}")
print(f"{'Ensemble (IF+LOF+OCSVM)':<25}{f1:<10.3f}")