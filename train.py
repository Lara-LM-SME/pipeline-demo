import sys
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")
mlflow.start_run()
print("MLflow tracking to local './mlruns' directory.")

data = {
    'text': ['love this product', 'terrible experience', 'works great', 'do not buy', 'amazing!', 'waste of money'],
    'sentiment': [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
X = df['text'].apply(lambda x: len(x.split())).to_frame(name='word_count')
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

C = 0.01
if len(sys.argv) > 1:
    try:
        C = float(sys.argv[1])
    except ValueError:
        print(f"Warning: invalid C '{sys.argv[1]}', using default {C}")

mlflow.log_param("C", C)
model = LogisticRegression(random_state=42, C=C)
model.fit(X_train, y_train)
print(f"Model trained with regularization C: {C}")

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
mlflow.log_metric("accuracy", accuracy)
print(f"Model Accuracy: {accuracy}")

if accuracy < 0.6:
    print("Validation Failed: Accuracy is below the 0.6 threshold.")
    mlflow.end_run(status='FAILED')
    sys.exit(1)
else:
    print("Validation Passed: Accuracy is sufficient.")
    import mlflow.sklearn
    mlflow.sklearn.log_model(model, "sentiment-model")
    mlflow.end_run()
    sys.exit(0)
