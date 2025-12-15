
# Credit Card Fraud Detection — End-to-End MLOps (Local)

![Architecture diagram](images/fe.png)


![Architecture diagram](MLflow.png)

## Overview

This project implements an end-to-end credit card fraud detection system designed to behave like a real machine learning service rather than a standalone model or notebook.

It covers the full lifecycle of an ML system, from data preparation and classification, through experiment tracking and model serving, to observability and operational constraints. While the system runs locally, the design choices mirror production environments closely, making it a practical foundation for cloud-native deployments.

The emphasis is on how data, models, APIs, and monitoring work together as a system, not just on achieving a particular accuracy score.

---

## Tech stack

This project is built using the following technologies:

* **Python 3.10**
  Used for data preparation, feature handling, model training, evaluation, and serving logic.

* **scikit-learn**
  Used to train a binary classification model for fraud detection.

* **pandas and NumPy**
  Used for data manipulation, preprocessing, transformation, and batch handling.

* **MLflow**
  Used to track experiments, parameters, metrics, and model artifacts, enabling reproducibility and traceability across training runs.

* **BentoML**
  Used to package the trained model and expose it as a production-style HTTP inference service with defined request and response schemas.

* **Streamlit**
  Used as a lightweight user interface to upload CSV files and trigger batch predictions.

* **Prometheus**
  Used to scrape inference-level metrics directly from the BentoML service.

* **Grafana**
  Used to visualise traffic patterns, latency, and service behaviour over time.

---

## Why I built this

This project started as a learning-focused exercise alongside ML and MLOps coursework.

The goal was to understand the full machine learning lifecycle in practice, not just how to train a model, but how data is collected and transformed, how models are evaluated and optimised, how they are deployed and served, and how operational concerns like scaling, governance, data protection, and access control come into play once a model is running.

Rather than treating these topics independently, the intent was to see how they connect end to end. How design decisions made early in data preparation affect model behaviour later, how models behave once exposed behind APIs, and how observability and reliability become just as important as predictive performance.

Building the system locally made it possible to explore these ideas in a controlled environment, while still structuring the project as if it were intended for production use.

This approach naturally led to building the project end to end and evolving it step by step, focusing on system behaviour, constraints, and trade-offs rather than isolated modelling tasks.

---

## What this project does

At a high level, this project builds a **binary fraud detection system** capable of scoring large batches of credit card transactions through an API.

The overall flow is:

1. A historical credit card transaction dataset is prepared and split into training and test sets
2. A binary classification model is trained and evaluated
3. Experiments, parameters, and metrics are tracked using MLflow
4. The trained model is registered and packaged using BentoML
5. A user uploads a CSV file through a UI
6. The data is sent to the BentoML service for inference
7. Fraud probabilities are returned and displayed
8. Inference traffic and latency are monitored via Prometheus and Grafana

The focus is not just on producing predictions, but on how the system behaves end to end under realistic usage patterns.

---

## Why batch inference

Fraud detection systems commonly need to score large volumes of transactions, for example:

* historical backfills
* delayed processing
* investigations
* audits

Serving only single-row predictions hides real-world constraints such as payload size, memory usage, request latency, and service saturation under load.

This project intentionally supports batch inference so those constraints are visible early. It makes performance, resource usage, and latency trade-offs explicit rather than abstract.

---

## Data and realism

The dataset used is a real, anonymised credit card transaction dataset with PCA-transformed features.

Although the feature names are abstract, the data structure reflects real fraud detection challenges:

* extreme class imbalance
* subtle separation between normal and fraudulent behaviour
* high cost of false positives
* probabilistic rather than deterministic decision-making

The model outputs **fraud probabilities**, not hard classifications. This mirrors how production systems operate, where thresholds are applied downstream based on risk tolerance, business rules, or human review processes.

---

## Model serving with BentoML

BentoML is used to ensure the model behaves like a service, not a script.

Using BentoML introduces considerations that are central to production systems, including:

* request and response schemas
* payload sizing and batch limits
* concurrency and execution behaviour
* API stability
* observability hooks

The service exposes:

* a `/predict` endpoint for batch inference
* a `/metrics` endpoint for monitoring

This allows the model to be treated as a first-class system component rather than a one-off artifact.

---

## Observability and monitoring

A key objective of this project was to make the system **observable**, not just functional.

Prometheus scrapes metrics directly from the BentoML service, including:

* request counts
* request latency
* in-flight requests
* runner execution timing

Grafana visualises these metrics over time, making it possible to answer questions such as:

* Is traffic reaching the model?
* How long do predictions take under different batch sizes?
* Are requests backing up or failing?
* How does service behaviour change under load?

This layer turns the model into something that can be reasoned about operationally, rather than treated as a black box.

---

## Security and governance considerations

Even in a local environment, the project is structured with production security and governance principles in mind.

These include:

* clear separation between training and serving components
* controlled access to inference via APIs
* awareness of data sensitivity and payload handling
* explicit boundaries around model exposure
* observability as a prerequisite for trust

While authentication, fine-grained access control, and cloud IAM are not implemented in this version, the architecture mirrors patterns that translate directly to secure, governed deployments in managed environments.

---

## Known limitations

The following limitations are documented and intentional:

* inference is synchronous
* batch size is constrained by memory and transport limits
* the model is trained once per run
* decision thresholds are not tuned for specific business objectives
* monitoring focuses on service health rather than model performance drift

Each of these constraints is understood and accounted for in the system design.

---

## Model lifecycle progression

As the system moved beyond basic training and serving, more attention was placed on how models are managed across their lifecycle rather than treated as static artefacts.

This includes clearly separating experimentation from models intended for downstream use, making model outputs traceable to the data and parameters that produced them, and ensuring that model behaviour can be observed and reasoned about once deployed.

At this stage, the project has progressed into the next phase of development, where the focus shifts from proving core data and serving plumbing to strengthening lifecycle discipline, operational confidence, and deployment readiness.

---

## Preparing for cloud deployment with local Kubernetes

Before introducing managed cloud services, the backend was prepared and validated using a local Kubernetes cluster.

This step was taken deliberately to reduce cost, limit external dependencies, and surface configuration issues early. Running the system under local orchestration made it possible to validate:

* container behaviour under orchestration
* service-to-service networking and connectivity
* configuration and environment management
* deployment and rollout mechanics

By addressing these concerns locally, the system is better positioned to transition into a managed environment such as EKS with fewer unknowns and tighter control over configuration and behaviour.
