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
matrix = pd.read_csv("../data/Event_occurrence_matrix.csv") if False else pd.read_csv("HDFS_v1/preprocessed/Event_occurrence_matrix.csv")
feature_cols = [col for col in matrix.columns if col.startswith("E")]
X = matrix[feature_cols]
y = (matrix["Label"] == "Fail").astype(int)

# --- Scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------
# Algorithm 1: Isolation Forest
# Unsupervised tree-based anomaly detection
# Anomalies = isolated in fewer splits (shallower)
# -----------------------------------------------
print("Training Isolation Forest...")
IF = IsolationForest(n_estimators=100, contamination=0.03, random_state=42, n_jobs=-1)
IF.fit(X_scaled)
if_scores = -IF.decision_function(X_scaled)  # higher = more anomalous

# -----------------------------------------------
# Algorithm 2: Local Outlier Factor (LOF)
# Density-based: compares local density of a point
# to its neighbors. Sparse = anomalous.
# Note: struggles with duplicate-heavy count data
# -----------------------------------------------
print("Training Local Outlier Factor...")
LOF = LocalOutlierFactor(n_neighbors=50, contamination=0.03, novelty=False, n_jobs=-1)
lof_preds = LOF.fit_predict(X_scaled)
lof_scores = -LOF.negative_outlier_factor_  # higher = more anomalous

# -----------------------------------------------
# Algorithm 3: One-Class SVM
# Learns a boundary around normal data.
# Anything outside boundary = anomaly.
# Uses 50K sample due to computational limits.
# -----------------------------------------------
print("Training One-Class SVM (this may take a few minutes)...")
sample_size = 50000
idx = np.random.RandomState(42).choice(len(X_scaled), sample_size, replace=False)
X_sample = X_scaled[idx]
OCSVM = OneClassSVM(kernel='rbf', nu=0.03)
OCSVM.fit(X_sample)
ocsvm_scores = -OCSVM.decision_function(X_scaled)  # higher = more anomalous

print("Building ensemble...")

# --- Normalize each score to 0-1 before combining ---
def normalize(scores):
    return (scores - scores.min()) / (scores.max() - scores.min())

if_norm   = normalize(if_scores)
lof_norm  = normalize(lof_scores)
ocsvm_norm = normalize(ocsvm_scores)

# -----------------------------------------------
# Weighted Ensemble Score
# IF gets highest weight — best individual performance
# LOF and OCSVM contribute as secondary signals
# -----------------------------------------------
ensemble_score = 0.6 * if_norm + 0.25 * lof_norm + 0.15 * ocsvm_norm

# --- Threshold at top 3% (contamination estimate) ---
threshold = np.percentile(ensemble_score, 97)
preds = (ensemble_score >= threshold).astype(int)

# --- Individual algorithm predictions for comparison ---
if_preds   = np.where(IF.predict(X_scaled) == -1, 1, 0)
lof_preds_binary = np.where(lof_preds == -1, 1, 0)

# --- Evaluate all three + ensemble ---
print("\n" + "=" * 50)
print(f"{'Method':<30}{'Precision':<12}{'Recall':<12}{'F1':<10}")
print("-" * 50)

for name, p in [("Isolation Forest", if_preds),
                ("Local Outlier Factor", lof_preds_binary),
                ("Ensemble (IF+LOF+OCSVM)", preds)]:
    print(f"{name:<30}"
          f"{precision_score(y, p):<12.3f}"
          f"{recall_score(y, p):<12.3f}"
          f"{f1_score(y, p):<10.3f}")

print("=" * 50)

# --- Confusion Matrix for Ensemble ---
cm = confusion_matrix(y, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("HDFS — Confusion Matrix (Ensemble: IF + LOF + OCSVM)")
plt.tight_layout()
plt.savefig("outputs/hdfs_ensemble_confusion_matrix.png")
print("\nConfusion matrix saved.")