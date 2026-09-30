import pandas as pd

# Check labels
labels = pd.read_csv("HDFS_v1/preprocessed/anomaly_label.csv")
print("=== LABELS ===")
print(labels.head())
print(f"Shape: {labels.shape}")
print(f"Anomaly counts:\n{labels['Label'].value_counts()}")

# Check event occurrence matrix
matrix = pd.read_csv("HDFS_v1/preprocessed/Event_occurrence_matrix.csv")
print("\n=== EVENT MATRIX ===")
print(matrix.head())
print(f"Shape: {matrix.shape}")