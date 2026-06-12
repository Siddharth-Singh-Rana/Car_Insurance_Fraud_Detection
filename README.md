# Car Insurance Fraud Detection using Machine Learning 🚗🕵️‍♂️

An end-to-end Machine Learning classification project designed to identify fraudulent automobile insurance claims. Using historical claim records, customer demographics, and incident details, the model detects suspicious patterns to help insurance companies mitigate heavy financial losses.

## 🎯 Objective & Business Problem
Insurance fraud costs companies billions of dollars annually. The main objective of this project is to automate the claim vetting process and accurately flag potentially fraudulent claims (`fraud_reported`). Since fraudulent cases are much fewer than legitimate ones, handling extreme **Class Imbalance** is a core focus of this analysis.

## 🛠️ Tech Stack & Libraries Used
- **Language:** Python
- **Data Preprocessing & EDA:** Pandas, NumPy, SciPy (Skewness handling)
- **Data Visualization:** Seaborn, Matplotlib (Correlation Heatmaps, Feature Distributions)
- **Machine Learning Workflow:** Scikit-Learn
- **Algorithms Evaluated:** Logistic Regression, Decision Tree Classifier, Random Forest Classifier

## 📊 Dataset Insights & Key Features
The predictive features used during model training include:
- **Customer Profile:** Age, Education Level, Socio-economic indicators.
- **Policy Details:** Policy Number, Umbrella Limit, Premium Amounts.
- **Incident Metrics:** Collision Type, Severity of Damage (Major/Minor), Authorities Informed, Number of Vehicles Involved.
- **Target Variable:** `fraud_reported` (Yes: 1, No: 0)

## 📈 Machine Learning Modeling & Results
- **Data Engineering:** Managed heavily skewed numerical features, encoded categorical variables systematically using `LabelEncoder`, and performed optimal train-test splitting.
- **Model Comparison & Performance:**
  - Evaluated multiple classifiers (Logistic Regression, Decision Tree, Random Forest).
  - **Key Finding:** Due to severe class imbalance, high raw accuracy was misleading. The **Decision Tree Classifier** out-performed other models in capturing true fraud instances (higher recall/precision tracking for Class 1).
  - **Evaluation Metrics:** Used Confusion Matrices and comprehensive Scikit-Learn `classification_report` metrics (Precision, Recall, F1-Score) to validate performance.

## 📂 Project Structure
- `fraud_detection.ipynb` - Complete Jupyter Notebook documenting step-by-step EDA, feature scaling, model training, and performance plots.
- `README.md` - Technical portfolio documentation.

## 💻 How to Run This Project Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/car-insurance-fraud-detection.git](https://github.com/YOUR_USERNAME/car-insurance-fraud-detection.git)
