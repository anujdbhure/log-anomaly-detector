import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load data ---
matrix = pd.read_csv("HDFS_v1/preprocessed/Event_occurrence_matrix.csv")

# --- Features: E1 to E29 ---
feature_cols = [col for col in matrix.columns if col.startswith("E")]
X = matrix[feature_cols]

# --- Labels: convert Success/Fail to 0/1 ---
y = (matrix["Label"] == "Fail").astype(int)

# --- Scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Isolation Forest ---
# contamination = anomaly ratio in HDFS (~0.03)
model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42, n_jobs=-1)
model.fit(X_scaled)

# --- Predict ---
preds = model.predict(X_scaled)
preds = np.where(preds == -1, 1, 0)

# --- Evaluate ---
p = precision_score(y, preds)
r = recall_score(y, preds)
f1 = f1_score(y, preds)

print("=" * 40)
print("HDFS — Isolation Forest Baseline")
print("=" * 40)
print(f"Precision: {p:.3f}")
print(f"Recall:    {r:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"\nTotal anomalies detected: {preds.sum()}")
print(f"Total actual anomalies:   {y.sum()}")

# --- Confusion Matrix ---
cm = confusion_matrix(y, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("HDFS — Confusion Matrix (Isolation Forest Baseline)")
plt.tight_layout()
plt.savefig("hdfs_confusion_matrix.png")
print("\nConfusion matrix saved.")