from flask import Flask, render_template, request
import pandas as pd
import requests
import base64
import io

app = Flask(__name__)

# DEMO ONLY: allow large file uploads (up to 200 MB)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

BENTOML_URL = "http://127.0.0.1:3000/predict"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file_data = request.form.get("file")

    if not file_data:
        return "No file uploaded", 400

    # Decode base64 CSV
    decoded = base64.b64decode(file_data.split(",")[1])

    # Load CSV into DataFrame
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    # Drop label column if present
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    # Send data to BentoML service
    response = requests.post(
        BENTOML_URL,
        json=df.to_dict(orient="records")
    )

    # Attach predictions
    fraud_probs = response.json()["fraud_probability"]
    df["fraud_probability"] = fraud_probs

    return render_template(
        "result.html",
        tables=[df.head(50).to_html(classes="data", header=True)]
    )

if __name__ == "__main__":
    app.run(port=5005, debug=True)
