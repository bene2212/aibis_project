# 🏦 Credit Risk Assessment & Customer Segmentation (XAI)

This repository contains the solution for a university project. It implements a Machine Learning pipeline to support "Passau Finance" in credit decision-making, featuring a fully interactive **Streamlit Dashboard** with Explainable AI (XAI) capabilities.

## 🎯 Project Goals

1.  **Credit Risk Assessment (Supervised Learning):**

    - Predict whether a loan applicant is a credit risk.
    - **Algorithm:** Random Forest Classifier.
    - **Strategy:** Optimization of the decision threshold to **0.30** to maximize **Recall** (minimizing missed defaults).
    - **Key Result:** Achieved \~85% Recall to prioritize bank safety over raw accuracy.

2.  **Customer Segmentation (Unsupervised Learning):**

    - Identify distinct customer personas for marketing.
    - **Algorithm:** K-Means Clustering ($k=4$) on PCA-reduced data.
    - **XAI:** Interpreting clusters using SHAP values.

3.  **Explainable AI (XAI):**

    - Implementation of **SHAP (SHapley Additive exPlanations)**.
    - Global explanations (Feature Importance) and Local explanations (Waterfall plots) for individual decisions.

4.  **Operational Dashboard:**

    - Deployment of models via a **Streamlit Web App**.
    - Simulates a real-time decision support system for executives and loan officers.

## 🛠️ Tech Stack

- **Python 3.11+**
- **Machine Learning:** `scikit-learn` (Random Forest, K-Means, PCA)
- **Explainability:** `shap`
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Deployment/App:** `streamlit`, `joblib`

## 📊 Dashboard Features

The dashboard is divided into three modules:

1.  **Executive Overview:** High-level KPIs, Confusion Matrix, ROC Curve, and Global SHAP analysis.
2.  **Customer Segmentation:** 2D Visualization of customer clusters (PCA) with persona descriptions.
3.  **Loan Officer Interface:** Live simulation tool where users can adjust applicant parameters (Age, Credit Amount, etc.) to get a real-time risk prediction and a local SHAP explanation.

## 🐳 Running with Docker

You can easily run the dashboard using Docker.

### Prerequisites

- Docker installed on your machine.
- Docker Compose (usually included with Docker Desktop).

### Steps

1.  **Build and Run**:
    Open a terminal in the project root and run:

    ```bash
    docker-compose up --build
    ```

2.  **Access the Dashboard**:
    Open your web browser and go to:
    [http://localhost:8501](http://localhost:8501)

3.  **Stop the Application**:
    Press `Ctrl+C` in the terminal or run:
    ```bash
    docker-compose down
    ```
