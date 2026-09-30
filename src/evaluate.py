import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# --- Load results ---
df = pd.read_csv("results.csv")

y_true = df["true_label"]
y_pred = df["predicted"]

# --- Calculate metrics ---
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("=" * 40)
print("MODEL EVALUATION")
print("=" * 40)
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")
print()
print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Log Anomaly Detection")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as confusion_matrix.png")

# --- Anomaly score distribution ---
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="anomaly_score", hue="true_label",
             bins=50, kde=True, palette=["blue", "red"])
plt.title("Anomaly Score Distribution (Normal vs Actual Anomaly)")
plt.xlabel("Anomaly Score (lower = more anomalous)")
plt.savefig("score_distribution.png")
print("Score distribution saved as score_distribution.png")

plt.show()