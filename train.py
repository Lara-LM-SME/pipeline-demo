import sys, os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ---------------- 1) MLflow ----------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.start_run()
print("MLflow tracking to local './mlruns' directory.")

# ---------------- 2) Load data ----------------
DATA_PATH = "data/reviews_fail.csv"

# ------------------------------------------------------------
# Use BOTH of these to explore CI pass/fail and fit quality:
# 1) DATA SOURCE: point to a big CSV (stable) or tiny CSV (fragile)
#    - Full:  export DATA_PATH=data/reviews.csv
#    - Tiny:  export DATA_PATH=data/reviews_fail.csv
# 2) FLEXIBILITY KNOB (C): bigger C = more flexible (risk: overfit),
#    smaller C = simpler (risk: underfit).
#    - Likely PASS: python train.py 1.0
#    - Likely FAIL: python train.py 1e-8   (esp. with tiny data or unigrams)
# ------------------------------------------------------------

assert os.path.exists(DATA_PATH), "Missing data/reviews.csv. Please add the CSV."

df = pd.read_csv(DATA_PATH)
assert {"text","sentiment"} <= set(df.columns), "reviews.csv must have text,sentiment columns"
print(f"Dataset size: {len(df)} (pos={df.sentiment.sum()}, neg={len(df)-df.sentiment.sum()})")

# ---------------- 3) Strict no-leak split ----------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.25, random_state=42, stratify=df["sentiment"]
)
print(f"Train size: {len(X_train_text)}, Test size: {len(X_test_text)}")

# ---------------- 4) Validation lever: "C" controls model flexibility ----------------
# Higher C  = higher cost for training mistakes -> less regularization -> more flexible model (overfit risk)
# Lower C  = lower  cost for mistakes          -> more regularization -> simpler model   (underfit risk)
# Tip: Use this along with DATA_PATH (full vs tiny) to force CI green/red on purpose for testing.
C = 1.0  # default, change this, run python train.py 0.5 for example
if len(sys.argv) > 1:
    try:
        C = float(sys.argv[1])  # e.g., python train.py 1.0  (likely PASS)
    except ValueError:
        print(f"Warning: invalid C '{sys.argv[1]}', using default {C}")
mlflow.log_param("C", C)  # keep runs auditable

# ---------------- 5) Train (fit vectorizer ONLY on train) ----------------
pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1,2), lowercase=True, min_df=1),
    LinearSVC(C=C, random_state=42)
)
pipe.fit(X_train_text, y_train)
preds = pipe.predict(X_test_text)

acc = accuracy_score(y_test, preds)
mlflow.log_metric("accuracy", acc)
print(f"Model Accuracy: {acc:.3f}")

# ---------------- 6) Gate ----------------
THRESH = 0.60
if acc < THRESH:
    print(f"Validation Failed: Accuracy is below the {THRESH} threshold.")
    mlflow.end_run(status="FAILED")
    sys.exit(1)
else:
    print("Validation Passed: Accuracy is sufficient.")
    import mlflow.sklearn
    mlflow.sklearn.log_model(pipe, "sentiment-model")
    mlflow.end_run()
    sys.exit(0)
