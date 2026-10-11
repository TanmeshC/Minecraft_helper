import os
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

TRAIN_PATH = os.path.join("data", "cleaned", "train.parquet")
TEST_PATH = os.path.join("data", "cleaned", "test.parquet")
MODEL_DIR = os.path.join("minementor", "ml", "classification")
METRICS_PATH = os.path.join("data", "processed", "difficulty_metrics.json")

def train_difficulty_model():
    print(f"Loading training data from {TRAIN_PATH}...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print(f"Train samples: {len(train_df):,}")
    print(f"Test samples:  {len(test_df):,}")

    X_train = train_df["instruction"].fillna("")
    y_train = train_df["difficulty"]

    X_test = test_df["instruction"].fillna("")
    y_test = test_df["difficulty"]

    print("Building TF-IDF + SGD Classifier pipeline for Difficulty...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=100,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Fitting difficulty classifier...")
    pipeline.fit(X_train, y_train)

    print("Evaluating difficulty on held-out test set...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"\n--- Difficulty Classifier Test Metrics ---")
    print(f"Accuracy:    {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "difficulty_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    metrics_summary = {
        "model_type": "TF-IDF + LogisticRegression(class_weight='balanced')",
        "test_samples": len(test_df),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": report
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Metrics report saved to {METRICS_PATH}")
    return metrics_summary

if __name__ == "__main__":
    train_difficulty_model()
