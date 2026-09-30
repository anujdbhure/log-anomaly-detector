from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

app = Flask(__name__)

def process_logs(df):
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Encode categorical columns
    level_encoder = LabelEncoder()
    event_encoder = LabelEncoder()
    df["level_encoded"] = level_encoder.fit_transform(df["level"])
    df["event_encoded"] = event_encoder.fit_transform(df["event"])

    # Time features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["seconds_since_last_log"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

    # Frequency features
    df["ip_frequency"] = df["ip"].map(df["ip"].value_counts())
    df["is_error"] = (df["level"] == "ERROR").astype(int)
    df["event_frequency"] = df["event"].map(df["event"].value_counts())

    # Burst detection
    df = df.set_index("timestamp")
    burst_counts = []
    for ip, group in df.groupby("ip"):
        counts = group.rolling("60s")["ip"].count()
        burst_counts.append(counts)
    df["same_ip_count_last_60s"] = pd.concat(burst_counts).sort_index()
    df = df.reset_index()

    feature_cols = [
        "level_encoded", "event_encoded", "hour",
        "seconds_since_last_log", "ip_frequency",
        "is_error", "same_ip_count_last_60s", "event_frequency"
    ]

    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_scaled)

    preds = model.predict(X_scaled)
    scores = model.decision_function(X_scaled)

    df["is_anomaly"] = np.where(preds == -1, 1, 0)
    df["anomaly_score"] = scores

    return df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        df = pd.read_csv(file, parse_dates=["timestamp"])
        df = process_logs(df)

        total = len(df)
        anomalies = int(df["is_anomaly"].sum())
        normal = total - anomalies

        # Prepare log table (last 50 for display)
        logs_display = df[["timestamp", "level", "ip", "event", "is_anomaly", "anomaly_score"]].copy()
        logs_display["timestamp"] = logs_display["timestamp"].astype(str)
        logs_display["anomaly_score"] = logs_display["anomaly_score"].round(4)
        logs_display = logs_display.sort_values("is_anomaly", ascending=False).head(100)

        return jsonify({
            "total": total,
            "anomalies": anomalies,
            "normal": normal,
            "logs": logs_display.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_hdfs", methods=["POST"])
def predict_hdfs():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        df = pd.read_csv(file)

        # HDFS event matrix format — E1 to E29
        feature_cols = [col for col in df.columns if col.startswith("E")]

        if len(feature_cols) == 0:
            return jsonify({"error": "No event columns (E1-E29) found. Make sure you upload Event_occurrence_matrix.csv"}), 400

        X = df[feature_cols]
        y = (df["Label"] == "Fail").astype(int) if "Label" in df.columns else None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(n_estimators=100, contamination=0.03, random_state=42, n_jobs=-1)
        model.fit(X_scaled)

        preds = model.predict(X_scaled)
        scores = model.decision_function(X_scaled)

        df["predicted"] = np.where(preds == -1, 1, 0)
        df["anomaly_score"] = np.round(scores, 4)

        total = len(df)
        anomalies = int(df["predicted"].sum())
        normal = total - anomalies

        # Build display table
        display_df = df[["BlockId", "Label", "predicted", "anomaly_score"]].copy() if "Label" in df.columns else df[["BlockId", "predicted", "anomaly_score"]].copy()
        display_df = display_df.sort_values("predicted", ascending=False).head(100)

        logs = []
        for _, row in display_df.iterrows():
            logs.append({
                "block_id": str(row["BlockId"]),
                "true_label": str(row["Label"]) if "Label" in row else "Unknown",
                "predicted": int(row["predicted"]),
                "anomaly_score": float(row["anomaly_score"])
            })

        return jsonify({
            "total": total,
            "anomalies": anomalies,
            "normal": normal,
            "logs": logs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)