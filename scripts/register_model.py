import bentoml
import mlflow
import mlflow.sklearn

# Load latest MLflow model from registry
model_name = "credit_card_fraud_model"
model_stage = "None"  # latest version regardless of stage

model_uri = f"models:/{model_name}/latest"

print(f"Loading model from MLflow URI: {model_uri}")

model = mlflow.sklearn.load_model(model_uri)

# Save model to BentoML
bento_model = bentoml.sklearn.save_model(
    name="credit_card_fraud_classifier",
    model=model,
    labels={
        "framework": "scikit-learn",
        "task": "binary-classification",
        "domain": "fraud-detection"
    }
)

print(" Model successfully registered in BentoML!! A Whole Lotta ...")
print(bento_model)
