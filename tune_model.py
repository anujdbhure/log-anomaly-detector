import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

X = pd.read_csv("features.csv")
y = pd.read_csv("labels.csv")

contamination_values = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]

print(f"{'Contamination':<15}{'Precision':<12}{'Recall':<12}{'F1':<12}")
print("-" * 50)

best_f1 = 0
best_c = None

for c in contamination_values:
    model = IsolationForest(n_estimators=100, contamination=c, random_state=42, n_jobs=-1)
    model.fit(X)
    pred = model.predict(X)
    pred = [1 if p == -1 else 0 for p in pred]

    p = precision_score(y, pred)
    r = recall_score(y, pred)
    f1 = f1_score(y, pred)

    print(f"{c:<15}{p:<12.3f}{r:<12.3f}{f1:<12.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_c = c

print(f"\nBest contamination: {best_c} (F1 = {best_f1:.3f})")