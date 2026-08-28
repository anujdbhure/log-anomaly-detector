import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Load the raw data ---
df = pd.read_csv("logs.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# --- Encode categorical columns ---
level_encoder = LabelEncoder()
event_encoder = LabelEncoder()
df["level_encoded"] = level_encoder.fit_transform(df["level"])
df["event_encoded"] = event_encoder.fit_transform(df["event"])

# --- Time-based features ---
df["hour"] = df["timestamp"].dt.hour
df["seconds_since_last_log"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

# --- IP frequency (global) ---
ip_counts = df["ip"].value_counts()
df["ip_frequency"] = df["ip"].map(ip_counts)

# --- NEW: is_error flag ---
df["is_error"] = (df["level"] == "ERROR").astype(int)
# --- NEW: event rarity feature ---
event_counts = df["event"].value_counts()
df["event_frequency"] = df["event"].map(event_counts)

# --- NEW: rolling burst count per IP (last 60 seconds) ---
df = df.set_index("timestamp")
burst_counts = []
for ip, group in df.groupby("ip"):
    counts = group.rolling("60s")["ip"].count()
    burst_counts.append(counts)
df["same_ip_count_last_60s"] = pd.concat(burst_counts).sort_index()
df = df.reset_index()

# --- Final feature set ---
feature_cols = [
    "level_encoded",
    "event_encoded",
    "hour",
    "seconds_since_last_log",
    "ip_frequency",
    "is_error",
    "same_ip_count_last_60s",
    "event_frequency"
]

X = df[feature_cols]
y = df["is_anomaly"]

# --- Scale features (important for Isolation Forest consistency) ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

X_scaled.to_csv("features.csv", index=False)
y.to_csv("labels.csv", index=False)

print("Preprocessing done.")
print(f"Feature shape: {X_scaled.shape}")
print(X_scaled.head(10))