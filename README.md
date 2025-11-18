# 🚀 End-to-End MLOps Pipeline for YouTube Comment Sentiment Analysis

This project implements a scalable, MLOps-driven pipeline for analyzing the sentiment of YouTube comments. It features an end-to-end workflow covering data versioning, iterative model experimentation (using MLflow), automated CI/CD deployment via GitHub Actions, and a high-performance Flask API for real-time inference.

-----

## ✨ Key Features & Technology Stack

This project demonstrates proficiency across the entire Machine Learning lifecycle, from feature engineering to production deployment.

### Machine Learning & NLP

  * **Chosen Model:** The final model selected for deployment is **LightGBM**, paired with the **ADASYN (Adaptive Synthetic Sampling)** technique.
  * **Classification:** Multi-class sentiment classification (Positive, Neutral, Negative).
  * **Feature Engineering:** Utilizes **TF-IDF** (Term Frequency-Inverse Document Frequency) with N-grams for text vectorization.
  * **Imbalance Handling:** Implementation of techniques like **ADASYN** to address class imbalance in the training data.
  * **Experimentation:** Comprehensive model comparison and hyperparameter tuning using various algorithms (LinearSVC, Logistic Regression, XGBoost, Naive Bayes) detailed in the `Notebooks/` directory.

### MLOps & Deployment

  * **Reproducibility (DVC):** Uses **Data Version Control (DVC)** (`dvc.yaml`) to version datasets, processed features, models, and metrics, ensuring experiments are fully reproducible.
  * **Experiment Tracking (MLflow):** Integrates **MLflow** for tracking model runs, logging parameters, metrics, and registering the final model in the Model Registry.
  * **Containerization (Docker):** The prediction service is containerized using **Docker** (`Dockerfile`) for consistent deployment across environments.
  * **CI/CD Pipeline:** A robust CI/CD workflow is set up with **GitHub Actions** (`ci-cd.yml`) to automate testing, Docker image building (ECR), and deployment to an **EC2** instance.
  * **RESTful API:** A high-performance inference API built with **Flask** and served by **Gunicorn**.

-----

## 🗃️ Dataset Source

The sentiment model was trained on a publicly available, labeled dataset sourced from **Reddit posts and comments**.

  * **Dataset Link:** [Reddit Sentimental analysis Dataset]([https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset](https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv))
  * **Rationale for using Reddit Data:** While the target domain is YouTube comments, Reddit data provides a highly valuable and practical training source because it shares several critical characteristics with YouTube comments: **informal language, high prevalence of internet slang, use of emojis/emoticons, and short, conversational sentence structure**. This cross-platform approach allows the model to learn general "social media English" sentiment features, which is highly effective and more practical than acquiring a massive, custom-labeled YouTube dataset.

-----

## 📊 Model Performance

The final deployed LightGBM model was evaluated on the held-out test set, achieving the following results:

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Test Accuracy** | `80.54%` | Overall correct classification rate. |
| **Test F1-Score (Weighted)** | `80.16%` | Balanced measure of precision and recall, crucial for multi-class problems. |
| **Test Precision (Weighted)** | `80.90%` | Weighted average of precision across all three classes. |
| **Test Recall (Weighted)** | `80.54%` | Weighted average of recall across all three classes. |
| **Cross-Validation F1 (Mean)** | `80.81%` | Average performance stability during training. |

-----

## 🔮 Future Improvements

To further enhance the accuracy, scalability, and robustness of this MLOps pipeline, the following improvements are planned:

### 1\. Model & Data Accuracy

  * **Deep Learning (Transfer Learning):** Replace the classical ML approach (LightGBM + TF-IDF) with a transformer-based model like **BERT** or **RoBERTa**. This would capture deeper contextual meaning in comments, potentially boosting accuracy significantly.
  * **Multilingual Support:** Implement **language detection** on incoming comments and switch to a **multilingual transformer model** (e.g., multilingual BERT) to accurately analyze non-English comments, significantly broadening the application's reach.
  * **Data Augmentation:** Implement techniques to generate synthetic comments or augment the minority classes to improve generalization, especially for sarcastic or nuanced language.

### 2\. MLOps & Production Robustness

  * **Automated Model Retraining:** Implement a scheduled DVC pipeline or a CI/CD job that automatically re-runs model training if the data drift score exceeds a set threshold.
  * **Data and Model Monitoring:** Integrate tools like **Prometheus** and **Grafana** to monitor:
      * **Data Drift:** Track changes in the distribution of incoming comments compared to training data.
      * **Model Performance Degradation:** Alert if the model's prediction accuracy or inference latency drops in production.
  * **Scalable Serving:** Migrate the API from a single EC2 instance to a managed container service like **AWS ECS/Fargate** or **Kubernetes** to allow for dynamic scaling based on real-time traffic demand.

-----

## ⚙️ Project Architecture

The pipeline follows a standard MLOps structure, managed by DVC stages and executed via the CI/CD workflow.

1.  **Data Management:** Raw data is stored and versioned using DVC.
2.  **Pipeline Execution:** The `dvc.yaml` defines stages: `data_ingestion`, `data_preprocessing`, `model_training`, and `model_evaluation`.
3.  **Experimentation:** Model training results, metrics, and plots are logged to MLflow.
4.  **Deployment:** The best model is promoted to the MLflow Model Registry (e.g., `Staging` or `Production`).
5.  **Service:** The Flask API dynamically loads the latest production-staged model from the MLflow Registry upon startup.
6.  **Automation:** GitHub Actions automatically builds and deploys the Docker image containing the Flask API service to AWS ECR/EC2 upon code merge to the main branch.

-----

## 💻 Local Setup & Development

### Prerequisites

  * Python 3.10+
  * Git, DVC (installed and initialized)
  * Docker (for containerization)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd youtube-sentiment-pipeline
    ```
2.  **Set up Python environment and dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Retrieve Data/Models (via DVC):**
    If the project is already linked to a remote DVC storage (e.g., S3), run:
    ```bash
    dvc pull
    ```
4.  **Run the DVC Pipeline:**
    Execute the entire MLOps pipeline defined in `dvc.yaml`:
    ```bash
    dvc repro
    ```

### Running the API Locally (with Docker)

To run the service locally, you must first ensure an MLflow Tracking Server is accessible.

1.  **Build the Docker Image:**
    ```bash
    docker build -t youtube-sentiment-api .
    ```
2.  **Run the Container:**
    You must provide the MLflow tracking URI as an environment variable for the API to load the model.
    ```bash
    docker run -d -p 8080:8080 \
        --name sentiment-service \
        -e MLFLOW_TRACKING_URI="http://[Your-MLflow-Server-IP]:5000/" \
        youtube-sentiment-api
    ```

-----

## 🧪 Testing

The project includes a comprehensive testing suite to ensure code quality and model integrity.

### Running Tests

All unit and integration tests are located in the `tests/` directory and can be executed using `pytest`.

```bash
# Run all tests and generate a coverage report
pytest --cov=src --cov=flask_api tests/
```

  * **Unit Tests:** Verify individual components like `data_preprocessing.py` and utility functions.
  * **Integration Tests:** Verify API endpoints (`/predict`, `/health`) function correctly by mocking the external MLflow dependencies.

-----

## 🌐 API Endpoints

The deployed Flask application exposes several endpoints for sentiment analysis and visualization.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service health check. |
| `GET` | `/` | Home status and deployed model info. |
| `POST` | `/predict` | Real-time sentiment prediction for a list of comments. |
| `POST` | `/batch_predict` | Batch prediction with aggregated sentiment statistics. |
| `POST` | `/generate_chart` | Generates a PNG pie chart of sentiment distribution. |
| `POST` | `/generate_wordcloud` | Generates a PNG word cloud from the provided comments. |

-----

## 📁 Project Structure

```
youtube-sentiment-pipeline/
├── .github/workflows/
│   └── ci-cd.yml             # GitHub Actions: CI/CD Pipeline (Test, Build, Deploy)
├── tests/                    # Unit and Integration Tests (for data, model, api)
├── data/
│   └── raw/                  # Raw dataset (tracked by DVC)
├── src/
│   ├── data/
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py # Handles cleaning and label remapping
│   └── model/
│       ├── model_building.py     # Training, MLflow logging
│       ├── model_evaluation.py   # Metric and plot generation
│       └── register_model.py     # Registers best model to MLflow Registry
├── flask_api/
│   └── app.py                # Flask API service loading model from MLflow
├── Notebooks/                # Extensive Jupyter Notebooks for EDA and Experimentation
├── dvc.yaml                  # DVC Pipeline definition
├── params.yaml               # Centralized parameters for DVC stages
├── Dockerfile                # Defines the API container image
└── requirements.txt
```
