import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATASET_PATH = "results/ml/immunotherapy_dataset.tsv"

df = pd.read_csv(DATASET_PATH, sep="\t")

required_cols = {"run_accession", "response"}
missing = required_cols - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns in dataset: {missing}")


patient_ids = [
    "SRR5930494",  
    "SRR5930536"    
]

feature_cols = [c for c in df.columns if c not in ["run_accession", "response"]]


tumor_demo = {
    patient_ids[0]: {
        "Tumor Mutation Burden": "High ",
        "PD-L1 Expression": "Moderate ",
        "CD8+ T-cell Infiltration": "Elevated "
    },
    patient_ids[1]: {
        "Tumor Mutation Burden": "Moderate ",
        "PD-L1 Expression": "Moderate ",
        "CD8+ T-cell Infiltration": ""
    }
}


print(" Demo")

print(f"Dataset size: {len(df)} patients")
print(f"Using patients: {patient_ids[0]} and {patient_ids[1]}")
print("Model: Gradient Based Classifer")
print("Threshold: 50% response probability")



for patient_id in patient_ids:
    if patient_id not in df["run_accession"].values:
        print(f"Patient {patient_id} not found in dataset.\n")
        continue

    train_df = df[df["run_accession"] != patient_id].copy()
    test_df = df[df["run_accession"] == patient_id].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["response"]

    X_test = test_df[feature_cols]
    actual = test_df["response"].iloc[0]

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=42
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[0]
    class_order = list(model.classes_)

    if "R" not in class_order:
        raise SystemExit("Model classes do not include 'R'.")

    response_prob = probs[class_order.index("R")]
    predicted = "R" if response_prob >= 0.5 else "NR"

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    patient_values = X_test.iloc[0]
    present_top = []
    for genus in importances.index:
        if patient_values[genus] > 0:
            present_top.append((genus, patient_values[genus], importances[genus]))
        if len(present_top) == 5:
            break

    print(f"Patient: {patient_id}")
    print(f"Actual Class: {actual}")
    print(f"Predicted Probability of Response: {response_prob*100:.1f}%")
    print(f"Predicted Class (50% threshold): {predicted}")
    print(f"Prediction Correct?: {'YES' if predicted == actual else 'NO'}")

    print("\nTumor biomarker panel:")
    for k, v in tumor_demo.get(patient_id, {}).items():
        print(f"  - {k}: {v}")

    print("\nTop informative microbiome features present in this patient:")
    if present_top:
        for genus, abundance, importance in present_top:
            print(f"  - {genus}: abundance={abundance:.2f}, model importance={importance:.4f}")
    else:
        print("  - No top-ranked genera with nonzero abundance found.")

    

print("Demo complete.")
