import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

TOTAL_NORMAL = 4750
TOTAL_ANOMALY = 250

ip_pool = [f"192.168.1.{i}" for i in range(1, 20)]
anomaly_ips = [f"10.0.0.{i}" for i in range(1, 5)]

events_normal = [
    "User logged in",
    "File accessed",
    "Session started",
    "Password changed",
    "Data exported",
]

events_anomaly = [
    "Failed login attempt",
    "Unauthorized access",
    "Port scan detected",
    "Multiple failed attempts",
    "Suspicious file access",
]

logs = []
start_time = datetime(2024, 1, 1, 0, 0, 0)

# Generate normal logs spread across the day
for i in range(TOTAL_NORMAL):
    logs.append({
        "timestamp": start_time + timedelta(seconds=random.randint(0, 86400)),
        "level": random.choice(['INFO', 'WARNING', 'ERROR']),
        "ip": random.choice(ip_pool),
        "event": random.choice(events_normal),
        "is_anomaly": 0
    })

# Generate anomaly logs in short bursts
for i in range(TOTAL_ANOMALY):
    logs.append({
        "timestamp": start_time + timedelta(seconds=random.randint(0, 3600)),
        "level": "ERROR",
        "ip": random.choice(anomaly_ips),
        "event": random.choice(events_anomaly),
        "is_anomaly": 1
    })

df = pd.DataFrame(logs)
df = df.sort_values("timestamp").reset_index(drop=True)
df.to_csv("logs.csv", index=False)

print(f"Generated {len(df)} logs")
print(f"Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
print(df.head(10))