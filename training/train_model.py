import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load processed data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]

X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

# Define pipeline
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        ),
    ]
)

# Start MLflow experiment
mlflow.set_experiment("credit_card_fraud_detection")

with mlflow.start_run():
    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)

    # Log metrics
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("average_precision", avg_precision)

    # Log model
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="credit_card_fraud_model"
    )

    print("ROC AUC:", roc_auc)
    print("Average Precision:", avg_precision)
