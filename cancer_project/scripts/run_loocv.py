import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

DATASET_PATH = "results/ml/immunotherapy_dataset.tsv"
RESULTS_PATH = "results/ml/loocv_results.tsv"
SUMMARY_PATH = "results/ml/loocv_summary.tsv"

df = pd.read_csv(DATASET_PATH, sep="\t")

required_cols = {"run_accession", "response"}
missing = required_cols - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

feature_cols = [c for c in df.columns if c not in ("run_accession", "response")]

print(f"Samples: {len(df)}")
print(f"Features: {len(feature_cols)}")
print(f"Class distribution: {df['response'].value_counts().to_dict()}")
print()

records = []

for i, row in df.iterrows():
    patient_id = row["run_accession"]
    actual = row["response"]

    train_df = df[df["run_accession"] != patient_id]
    test_row = df[df["run_accession"] == patient_id]

    X_train = train_df[feature_cols].values
    y_train = train_df["response"].values
    X_test = test_row[feature_cols].values

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[0]
    class_order = list(model.classes_)
    response_prob = probs[class_order.index("R")]
    predicted = "R" if response_prob >= 0.5 else "NR"

    records.append({
        "run_accession": patient_id,
        "actual": actual,
        "predicted_prob_R": round(response_prob, 4),
        "predicted_class": predicted,
        "correct": "YES" if predicted == actual else "NO",
    })

    status = "CORRECT" if predicted == actual else "WRONG"
    print(f"  {patient_id}  actual={actual}  pred={predicted}  prob_R={response_prob:.3f}  [{status}]")

results_df = pd.DataFrame(records)

y_true = results_df["actual"].values
y_pred = results_df["predicted_class"].values
y_prob = results_df["predicted_prob_R"].values

accuracy = (y_true == y_pred).mean()
auc = roc_auc_score((y_true == "R").astype(int), y_prob)

labels = ["NR", "R"]
cm = confusion_matrix(y_true, y_pred, labels=labels)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

print()
print("=" * 50)
print("LOOCV SUMMARY")
print("=" * 50)
print(f"Accuracy:    {accuracy:.4f}  ({int(accuracy * len(results_df))}/{len(results_df)})")
print(f"ROC-AUC:     {auc:.4f}")
print(f"Sensitivity: {sensitivity:.4f}  (TP={tp}, FN={fn})")
print(f"Specificity: {specificity:.4f}  (TN={tn}, FP={fp})")
print()
print("Confusion matrix (rows=actual, cols=predicted):")
print(f"             pred_NR  pred_R")
print(f"  actual_NR    {tn:3d}      {fp:3d}")
print(f"  actual_R     {fn:3d}      {tp:3d}")
print()

os.makedirs("results/ml", exist_ok=True)
results_df.to_csv(RESULTS_PATH, sep="\t", index=False)
print(f"Per-patient results saved to {RESULTS_PATH}")

summary = pd.DataFrame([{
    "n_samples": len(results_df),
    "n_R": int((y_true == "R").sum()),
    "n_NR": int((y_true == "NR").sum()),
    "accuracy": round(accuracy, 4),
    "roc_auc": round(auc, 4),
    "sensitivity": round(sensitivity, 4),
    "specificity": round(specificity, 4),
    "TP": int(tp),
    "FP": int(fp),
    "TN": int(tn),
    "FN": int(fn),
}])
summary.to_csv(SUMMARY_PATH, sep="\t", index=False)
print(f"Summary metrics saved to {SUMMARY_PATH}")
