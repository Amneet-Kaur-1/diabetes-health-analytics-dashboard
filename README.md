
# Diabetes Health Analytics Dashboard

An interactive machine learning dashboard built with Python and Streamlit using the CDC Diabetes Health Indicators dataset.

## Dataset

**Dataset name:** CDC Diabetes Health Indicators

**Source:** [UCI Machine Learning Repository — CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

The dataset contains 253,680 records and 21 features relating to health indicators, lifestyle, and diabetes status. It is derived from CDC survey data. [Source](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). 

The dataset is used for educational and machine learning research purposes.

## Project Overview

This project explores health indicators associated with diabetes and demonstrates how machine learning can support data analysis and risk classification.

The dashboard includes:

- **Overview:** Summary of the dataset and project.
- **EDA Explorer:** Explore distributions and relationships between health indicators.
- **Model Comparison:** Compare the performance of different machine learning models.
- **Feature Insights:** Explore the features that contribute to the trained model.
- **Evaluation:** View model evaluation results, including confusion matrix and ROC/precision-recall curves.
- **Prediction Demo:** Try a sample prediction using health indicator inputs, if enabled in the current app.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Plotly
- Streamlit
- Joblib

## Project Structure

```text
machine-learning-model/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── diabetes_random_forest.joblib
└── data/
    ├── confusion_matrix.csv
    ├── evaluation_summary.csv
    ├── model_comparison.csv
    ├── precision_recall_curve.csv
    └── roc_curve.csv
```

The raw dataset is excluded from this repository for now. Follow the dataset source and usage terms before downloading or redistributing it.

## Run Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Replace the placeholders with your repository URL and folder name.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the permitted dataset file at the location expected by `app.py`:

`data/cdc_diabetes_health_indicators.csv`

### 5. Launch the dashboard

```bash
streamlit run app.py
```

## Notes

- The dashboard is an educational data analytics project, not a medical diagnostic tool.
- Model outputs should not be used to make medical decisions.
- Ensure the dataset and model files are available before running the dashboard.
