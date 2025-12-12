# 🚀 End-to-End MLOps Pipeline & Chrome Extension for YouTube Sentiment Analysis

This project implements a scalable, MLOps-driven pipeline for analyzing the sentiment of YouTube comments, now featuring a **custom Google Chrome Extension** for real-time insights.

The system covers the entire lifecycle: data versioning, iterative model experimentation (MLflow), automated CI/CD deployment, a high-performance Flask API, and a user-friendly browser plugin that visualizes sentiment trends directly on YouTube video pages.

[![Watch the video](https://img.youtube.com/vi/fI2g6Cocwwo/maxresdefault.jpg)](https://www.youtube.com/watch?v=fI2g6Cocwwo)


---

## ✨ Key Features & Technology Stack

### 🧩 **Chrome Extension (Frontend)**

- **Real-time Analysis:** Instantly fetches and analyzes comments from the active YouTube video tab.
- **Interactive Dashboard:** A modern popup UI displaying:
  - **Overall Sentiment Score:** A normalized 0-10 score with confidence metrics.
  - **Visualizations:** Dynamic Pie Charts, Sentiment Trend Graphs (over time), and Word Clouds.
  - **Top Comments:** Highlights specific comments categorized by sentiment (Positive, Neutral, Negative).
- **Tech:** JavaScript (ES6+), HTML5, CSS3 (Dark Mode), Chrome Scripting API.

### 🧠 **Machine Learning & NLP**

- **Model:** **LightGBM** classifier optimized with **ADASYN** (Adaptive Synthetic Sampling) for class imbalance.
- **Performance:** ~80% Accuracy & F1-Score on social media text.
- **Feature Engineering:** **TF-IDF** vectorization with N-grams and custom text preprocessing.
- **Experimentation:** Benchmarked against LinearSVC, Logistic Regression, and XGBoost using `Notebooks/`.

### ⚙️ **MLOps & Backend**

- **Reproducibility (DVC):** Dataset and pipeline versioning using **DVC** (`dvc.yaml`).
- **Tracking (MLflow):** Experiment tracking and Model Registry for versioned model management.
- **Containerization:** **Dockerized** Flask API for consistent deployment.
- **CI/CD:** **GitHub Actions** pipeline for automated testing, building, and deploying to **AWS EC2**.
- **API:** RESTful Flask service serving predictions and generating visualization images on-the-fly.

---

## 🏗️ Project Architecture

The solution consists of two main components: the **Backend MLOps Pipeline** and the **Frontend Extension**.

1.  **Data & Training:** DVC manages data ingestion/processing. Models are trained and logged to MLflow.
2.  **Deployment:** GitHub Actions deploys the best model (via Docker) to an AWS EC2 instance.
3.  **Inference:**
    - **User** opens a YouTube video and clicks the Extension icon.
    - **Extension** scrapes comments using the YouTube Data API.
    - **Extension** sends comments to the **Flask API** (`/predict_with_timestamps`).
    - **API** returns sentiment predictions and generates charts (Pie/Trend/WordCloud).
    - **Extension** renders the dashboard for the user.

---

## 📊 Model Performance

The deployed **LightGBM + ADASYN** model achieves robust performance on social media text:

| Metric                  | Value    | Notes                                      |
| :---------------------- | :------- | :----------------------------------------- |
| **Test Accuracy**       | `79.12%` | Overall classification correctness.        |
| **F1-Score (Weighted)** | `78.73%` | Balanced precision/recall for all classes. |
| **Precision**           | `79.70%` | High reliability in positive prediction.   |
| **Cross-Validation F1** | `79.60%` | Stable performance across data folds.      |

---

## 📁 Project Structure

```bash
youtube-sentiment-pipeline/
├── .github/workflows/
│   └── ci-cd.yml                 # CI/CD: Test, Build, Deploy to EC2
├── Youtube_Sentiment_Extension/  # 🆕 Chrome Extension Source Code
│   ├── manifest.json             # Extension configuration
│   ├── popup.html                # Dashboard UI
│   ├── popup.js                  # Logic for fetching comments & calling API
│   └── icons/                    # Extension icons
├── flask_api/
│   └── app.py                    # Inference API (Predictions + Image Gen)
├── src/                          # Source code for training pipeline
│   ├── data/                     # Ingestion & Preprocessing
│   └── model/                    # Training, Evaluation & Registration
├── Notebooks/                    # EDA & Model Experiments
├── dvc.yaml                      # DVC Pipeline definition
├── params.yaml                   # Hyperparameters
├── Dockerfile                    # API Container definition
└── requirements.txt
```

---

## 💻 Installation & Setup

### 1\. Backend Setup (Local)

To run the model training or API locally:

```bash
# Clone repo
git clone [Your-Repo-URL]
cd youtube-sentiment-pipeline

# Install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run MLOps pipeline (Reproduction)
dvc repro

# Run API locally
python flask_api/app.py
```

### 2\. Chrome Extension Setup

To use the plugin in your browser:

1.  Open Chrome and navigate to `chrome://extensions/`.
2.  Toggle **Developer mode** (top right corner).
3.  Click **Load unpacked**.
4.  Select the `Youtube_Sentiment_Extension` folder from this repository.
5.  Open any YouTube video and click the extension icon to see insights\!

> **Note:** The extension is configured to hit the AWS EC2 endpoint by default. To use localhost, update `API_URL` in `popup.js`.

---

## 🌐 API Endpoints

The Flask API provides the following endpoints for the extension:

| Method | Endpoint                   | Description                                                |
| :----- | :------------------------- | :--------------------------------------------------------- |
| `POST` | `/predict_with_timestamps` | Predicts sentiment for a list of comments with timestamps. |
| `POST` | `/batch_predict`           | Batch prediction with aggregated stats.                    |
| `POST` | `/generate_chart`          | Returns a PNG Pie Chart of sentiment distribution.         |
| `POST` | `/generate_trend_graph`    | Returns a PNG Line Graph of sentiment over time.           |
| `POST` | `/generate_wordcloud`      | Returns a PNG Word Cloud of frequent terms.                |
| `GET`  | `/health`                  | Service health check.                                      |
| `GET`  | `/model_info`              | Returns metadata about the currently loaded MLflow model.  |

---

## 🔮 Future Improvements

- **Transformer Models:** Upgrade from LightGBM to **BERT/RoBERTa** for better context understanding.
- **Multilingual Support:** Add language detection to handle non-English comments.
- **Live Streaming Support:** Adapt the extension to work with YouTube Live Chat.
- **User Feedback Loop:** Allow users to correct sentiment labels in the extension to retrain the model (Active Learning).

<!-- end list -->

```

```
