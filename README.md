# Predict Remaining Useful Life (RUL) via Machine Learning

![RUL Dashboard](generated-icon.png)

## Overview

This project is an end-to-end machine learning pipeline and interactive dashboard for predicting the **Remaining Useful Life (RUL)** of machinery using regression models. Built with Python, Streamlit, scikit-learn, and TensorFlow/Keras, it supports flexible data ingestion, advanced feature engineering, model training, evaluation, and monitoring—all in a user-friendly web app.

---

## 🚀 Features

- **Flexible Data Ingestion**
  - Load data from NASA CMAPSS, AWS S3, REST APIs, or use built-in sample data.
- **Comprehensive Preprocessing & Feature Engineering**
  - Handles missing values, rolling statistics, difference features, and scaling.
- **Exploratory Data Analysis (EDA)**
  - Visualize sensor trends, feature distributions, and summary statistics.
- **Model Training**
  - Train and compare Random Forest, Gradient Boosting, and Deep CNN models.
  - Hyperparameter tuning for optimal performance.
- **Model Evaluation**
  - Metrics: RMSE, MAE, R², NASA Score, PHM21 Score.
  - Visualizations: Predicted vs Actual, error distributions, feature importance.
- **Monitoring Dashboard**
  - Track model performance and data drift over time.
- **AWS S3 Integration**
  - Seamlessly load and save data/models to S3 buckets.
- **Modular & Extensible**
  - Clean, well-organized codebase ready for further MLOps, deployment, and automation.

---

## 🏗️ Project Structure

├── app.py # Main Streamlit app
├── pages/ # Streamlit multipage UI (EDA, Training, Evaluation, Monitoring)
├── utils/ # Data processing, feature engineering, AWS, deep learning, metrics
├── models/ # Saved model files
├── data/ # NASA CMAPSS and sample datasets
├── .streamlit/ # Streamlit config
├── pyproject.toml # Project dependencies
├── uv.lock # Locked dependency versions


---

## 📊 Example Screenshots

- **Data Exploration:** Visualize sensor data and summary stats.
- **Model Training:** Train and tune multiple models.
- **Evaluation:** Analyze predictions, errors, and feature importances.
- **Monitoring:** Track model performance over time.

---

## 🛠️ How to Run

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Predict-Remaining-Useful-Life--RUL--via-regression
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   uv pip install .
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧠 Project Phases & Capabilities

- **Phase 1: Planning & Requirements**  
  - Predict RUL, use RMSE as a key metric, support multiple data sources.
- **Phase 2: Data Ingestion & Storage**  
  - NASA CMAPSS, AWS S3, API, and sample data support.
- **Phase 3: Preprocessing & Feature Engineering**  
  - Rolling stats, diffs, scaling, missing value handling.
- **Phase 4: EDA**  
  - Interactive visualizations and summary statistics.
- **Phase 5: Model Development & Training**  
  - Random Forest, Gradient Boosting, Deep CNN, hyperparameter tuning.
- **Phase 6: Evaluation**  
  - RMSE, MAE, R², NASA/PHM21 scores, error analysis, feature importance.
- **Phase 7: Monitoring**  
  - Local dashboard for performance and drift (AWS CloudWatch ready).
- **Phase 8-10: (Ready for Extension)**  
  - Modular codebase for easy integration with SageMaker, CI/CD, and MLOps.

---

## 🌟 Future Work

- **Full AWS MLOps Integration:**  
  Add SageMaker Pipelines, Model Monitor, and EventBridge automation.
- **API Deployment:**  
  Serve models via REST API (Flask/FastAPI) and/or SageMaker endpoints.
- **Automated Retraining & Feedback Loops**
- **Unit/Integration Tests & CI/CD Pipeline**
- **Expanded Documentation & Tutorials**

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👤 Author

**Ankit Addya**  
*Data Science Enthusiast*

---

## 📬 Contact

- [LinkedIn](www.linkedin.com/in/ankit-addya-6a6b14159)
---

> **Built as a showcase of end-to-end ML engineering, data science, and MLOps skills.**