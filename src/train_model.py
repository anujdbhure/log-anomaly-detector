import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# --- Load preprocessed data ---
X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv")

# --- Initialize Isolation Forest ---
model = IsolationForest(
    n_estimators=100,      # number of trees in the forest
    contamination=0.05,    # our expected anomaly ratio (5%, matches our data)
    random_state=42,       # for reproducible results
    n_jobs=-1               # use all CPU cores
)

# --- Train the model (unsupervised - we never pass y here) ---
model.fit(X)

# --- Get predictions ---
# -1 = anomaly, 1 = normal (sklearn's convention)
predictions = model.predict(X)

# --- Get anomaly scores (lower = more anomalous) ---
scores = model.decision_function(X)

# --- Convert predictions to match our labels (1 = anomaly, 0 = normal) ---
df_results = X.copy()
df_results["true_label"] = y
df_results["predicted"] = np.where(predictions == -1, 1, 0)
df_results["anomaly_score"] = scores

# --- Save results ---
df_results.to_csv("results.csv", index=False)

# --- Save the trained model for later use ---
joblib.dump(model, "isolation_forest_model.pkl")

print("Model trained successfully.")
print(f"Total anomalies detected: {df_results['predicted'].sum()}")
print(f"Total actual anomalies: {df_results['true_label'].sum()}")
print(df_results.head(10))