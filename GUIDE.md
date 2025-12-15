
# Credit Card Fraud Detection — End-to-End MLOps Project

## Overview

This project builds an end-to-end **fraud detection MLOps pipeline** using a real-world sized dataset.
It covers data preparation, model training and evaluation, model serving, a user interface for predictions, and observability using Prometheus and Grafana.

The focus is not on inventing a novel fraud algorithm, but on **building a system that works end to end**, handles real data sizes, exposes meaningful metrics, and surfaces the operational issues you encounter in practice.

---

## Tech stack

* **Python 3.10**
* **Pandas, NumPy, scikit-learn**
* **MLflow** for experiment tracking and model registry
* **BentoML** for model serving
* **Streamlit** for UI
* **Prometheus** for metrics scraping
* **Grafana** for visualisation
* **Git & GitHub** for version control

---

## Project structure (final)

```
credit-card-fraud-project/
│
├── app/
│   ├── app.py
│   ├── streamlit_app.py
│   └── templates/
│
├── data/
│   ├── raw/              # ignored by git
│   └── processed/        # ignored by git
│
├── training/
│   └── train_model.py
│
├── serving/
│   └── service.py
│
├── scripts/
│   └── register_model.py
│
├── diagram/
│   └── architecture.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

Large datasets and MLflow artifacts are **deliberately excluded from version control**.

---

## Step 1 — Environment setup

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Verify Python version:

```powershell
python --version
```

Python **3.10.x** is used to maintain compatibility with MLflow and BentoML.

---

## Step 2 — Data preparation

Place the raw dataset in:

```
data/raw/creditcard.csv
```

Run the preparation script:

```powershell
python data/prepare_data.py
```

What this does:

* Loads the raw dataset
* Drops the target column from features
* Splits into train and test sets
* Writes CSVs to `data/processed/`

Expected output:

* Class imbalance visible (fraud ≈ 0.17%)
* Feature columns dominated by floating-point PCA values

This is **expected and correct**.

---

## Step 3 — Model training and MLflow tracking

Train the model:

```powershell
python training/train_model.py
```

What happens:

* Model trains using scikit-learn
* Metrics such as ROC-AUC and average precision are logged
* Artifacts are stored in `mlruns/`
* Model is registered to MLflow

Optional UI check:

```powershell
mlflow ui
```

Then open:

```
http://127.0.0.1:5000
```

---

## Step 4 — Model registration

Register the trained model:

```powershell
python scripts/register_model.py
```

This promotes the model into the MLflow registry so it can be consumed by BentoML.

---

## Step 5 — Model serving with BentoML

Start the BentoML service:

```powershell
bentoml serve serving.service:svc --reload --port 3001
```

Important details:

* Service listens on **port 3001**
* `/predict` endpoint accepts JSON records
* `/metrics` endpoint exposes Prometheus metrics

Verify service is running:

```
http://localhost:3001
http://localhost:3001/metrics
```

---

## Step 6 — Frontend with Streamlit

Run the Streamlit app:

```powershell
streamlit run app/streamlit_app.py
```

What it does:

* Allows CSV upload
* Drops `Class` column if present
* Sends data in chunks to BentoML
* Displays fraud probability per row

This avoids request size limits and mirrors real batch scoring workflows.

---

## Step 7 — Observability with Prometheus

Run Prometheus with explicit config path:

```powershell
C:\monitoring\prometheus\prometheus.exe --config.file=C:\monitoring\prometheus\prometheus.yml
```

Prometheus listens on:

```
http://localhost:9090
```

Confirm BentoML target is **UP** and scraping `/metrics`.

---

## Step 8 — Grafana dashboards

Start Grafana:

```powershell
grafana-server.exe
```

Grafana UI:

```
http://localhost:3000
```

Configure:

* Add Prometheus data source (`http://localhost:9090`)
* Create panels using BentoML metrics such as:

  * `bentoml_api_server_request_total`
  * `bentoml_api_server_request_duration_seconds`

---

## Step 9 — Version control and GitHub

Initialise Git:

```powershell
git init
git branch -M master
```

Add files:

```powershell
git add .
git commit -m "Initial commit: credit card fraud detection MLOps pipeline"
```

Add remote:

```powershell
git remote add origin https://github.com/Y2not2nd/ML-Credit-Card-Fraud-Detector.git
```

Push:

```powershell
git push -u origin master
```

Later create `main` if required:

```powershell
git checkout -b main
git push -u origin main
```

---

## Issues encountered and how they were solved

### 1. Python version mismatch

**Issue:** Python 3.13 caused incompatibilities
**Solution:** Installed and used Python 3.10 explicitly

---

### 2. Missing `data/processed` directory

**Issue:** Pandas failed to write CSV files
**Solution:** Created directory before writing output

---

### 3. CSV uploads too large for HTTP requests

**Issue:** `Request Entity Too Large` errors
**Solution:** Switched to Streamlit and chunked uploads

---

### 4. BentoML deprecation warnings

**Issue:** Deprecated runners and IO APIs
**Solution:** Accepted warnings for now, pinned compatible versions

---

### 5. BentoML `predict_proba` attribute error

**Issue:** Runner did not expose `predict_proba`
**Solution:** Called model method correctly via runner API

---

### 6. Missing `fraud_probability` key

**Issue:** Frontend expected wrong JSON structure
**Solution:** Aligned BentoML response schema with frontend

---

### 7. Port conflicts (3000, 3001, 9090)

**Issue:** Multiple services bound to same ports
**Solution:** Explicitly assigned ports and verified with logs

---

### 8. Prometheus config path errors

**Issue:** Prometheus couldn’t find config file
**Solution:** Passed absolute path via `--config.file`

---

### 9. GitHub file size limits

**Issue:** CSVs and MLflow artifacts exceeded 100MB
**Solution:** Removed from Git, added to `.gitignore`, recommitted clean history

---

### 10. Branch confusion (`master` vs `main`)

**Issue:** Work committed to wrong branch
**Solution:** Reinitialised repo cleanly and pushed correct branch

---

## Final note

This project reflects what actually happens when building ML systems:

* real datasets
* real infrastructure limits
* real tooling friction

The value is not just the trained model, but the **understanding of how the pieces fit together** and where systems fail if you don’t respect scale, interfaces, and observability.

This is a complete, end-to-end baseline that can now evolve into:

* scheduled training
* automated CI/CD
* cloud deployment
* model version comparison in production

When you’re ready, the next iteration should focus on **automation and deployment**, not new algorithms.
