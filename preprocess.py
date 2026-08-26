import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- Load the raw data ---
df = pd.read_csv("logs.csv", parse_dates=["timestamp"])

# --- 1. Sort by timestamp (needed for time-gap feature) ---
df = df.sort_values("timestamp").reset_index(drop=True)

# --- 2. Encode categorical columns ---
level_encoder = LabelEncoder()
event_encoder = LabelEncoder()

df["level_encoded"] = level_encoder.fit_transform(df["level"])
df["event_encoded"] = event_encoder.fit_transform(df["event"])

# --- 3. Extract time-based features ---
df["hour"] = df["timestamp"].dt.hour

# seconds since previous log entry (burst detection)
df["seconds_since_last_log"] = df["timestamp"].diff().dt.total_seconds()
df["seconds_since_last_log"] = df["seconds_since_last_log"].fillna(0)

# --- 4. IP frequency feature ---
ip_counts = df["ip"].value_counts()
df["ip_frequency"] = df["ip"].map(ip_counts)

# --- 5. Select final features for the model ---
feature_cols = [
    "level_encoded",
    "event_encoded",
    "hour",
    "seconds_since_last_log",
    "ip_frequency"
]

X = df[feature_cols]
y = df["is_anomaly"]  # kept aside ONLY for evaluation later

# --- Save processed data ---
X.to_csv("features.csv", index=False)
y.to_csv("labels.csv", index=False)

print("Preprocessing done.")
print(f"Feature shape: {X.shape}")
print(X.head(10))