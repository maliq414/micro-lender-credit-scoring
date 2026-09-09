# 💳 Micro-Lender Credit Scoring

## Machine Learning for Credit Default Risk Assessment

A machine-learning credit-risk decision-support prototype for estimating borrower default risk using credit limits, repayment behaviour, billing history, and payment history.

## 🚀 Live Application

**Try the deployed application:**

https://micro-lender-credit-risk.streamlit.app/

The application produces:

- Default probability
- Credit score (0–100)
- Risk category
- Lending recommendation
- Supporting risk indicators

## 📌 Project Overview

This 3MTT Data Science capstone develops a machine-learning prototype for estimating borrower default risk.

The project compares Logistic Regression and Random Forest models. The selected Random Forest model converts predicted default probability into an interpretable credit score and risk category and is deployed through Streamlit.

## 🎯 Objectives

- Predict borrower default probability.
- Compare machine-learning classification models.
- Identify important repayment and financial risk indicators.
- Convert probability into a 0–100 credit score.
- Classify borrowers as Low, Medium, or High Risk.
- Generate lending decision-support recommendations.
- Deploy the final model as an interactive web application.

## 📊 Dataset

The project uses the **UCI Default of Credit Card Clients dataset**, containing 30,000 historical customer records.

The model uses financial and repayment-behaviour information including credit limits, repayment status, monthly bills, and monthly payments.

## 🛠️ Feature Engineering

Two additional behavioural indicators were created:

**DELAY_MONTHS** — Number of the previous six repayment-status periods with a positive repayment delay.

**CREDIT_UTILIZATION** — Calculated as:

`BILL_AMT1 / LIMIT_BAL`

The final model uses **21 financial and repayment-behaviour features**.

## 🤖 Machine Learning Models

Two classification models were evaluated:

1. Logistic Regression
2. Random Forest

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.766 | 0.475 | 0.560 | 0.514 | 0.747 |
| **Random Forest** | **0.779** | **0.501** | **0.578** | **0.537** | **0.774** |

Random Forest was selected as the final model because it achieved stronger overall predictive performance, including the highest ROC-AUC and F1 score.

## 🔍 Important Risk Indicators

Random Forest feature importance showed that repayment behaviour was especially influential.

Leading features included:

- `PAY_0`
- `DELAY_MONTHS`
- `PAY_2`
- `PAY_3`
- `CREDIT_UTILIZATION`
- `PAY_4`
- `PAY_AMT1`
- `LIMIT_BAL`
- `BILL_AMT1`
- `PAY_AMT2`

These results suggest that recent repayment behaviour is particularly useful for identifying elevated default risk in this dataset.

## 💯 Credit Risk Scoring

The predicted Random Forest default probability is transformed into a 0–100 credit score:

`Credit Score = (1 - Default Probability) × 100`

Higher scores represent lower estimated default risk.

| Credit Score | Risk Category | Recommendation |
|---:|---|---|
| 70–100 | 🟢 Low Risk | Consider for Approval |
| 40–69 | 🟡 Medium Risk | Manual Review |
| 0–39 | 🔴 High Risk | Further Assessment Required |

## 📈 Risk Category Validation

The risk categories were evaluated against actual default outcomes in the held-out test set.

| Risk Category | Borrowers | Observed Default Rate |
|---|---:|---:|
| Low Risk | 2,472 | 8.09% |
| Medium Risk | 2,394 | 20.18% |
| High Risk | 1,127 | 57.05% |

The increasing observed default rate across the three groups demonstrates useful risk separation in the test sample.


## 🌐 Streamlit Application

The selected Random Forest model is deployed as an interactive Streamlit web application.

Users can provide credit limit, six months of repayment status, six months of bill amounts, and six months of payment amounts.

The application automatically calculates `DELAY_MONTHS` and `CREDIT_UTILIZATION` before generating the prediction.

### 🚀 Live Demo

https://micro-lender-credit-risk.streamlit.app/

The application returns:

- Default probability
- Credit score
- Risk category
- Lending recommendation
- Credit utilization
- Number of repayment-delay months

## 📁 Repository Structure

The repository contains:

- `app/` — Streamlit application and trained model
- `notebooks/` — Data science and machine-learning notebook
- `README.md` — Project documentation
- `requirements.txt` — Project dependencies
- `.gitignore` — Git configuration

The `app/` directory contains:

- `app.py`
- `behavior_features.pkl`
- `micro_lender_random_forest.pkl`
- `requirements.txt`

## 💻 Running the Application Locally

Clone the GitHub repository, install the packages listed in `app/requirements.txt`, and start the application with:

`streamlit run app/app.py`

## ⚠️ Responsible Use and Limitations

This project is an **educational machine-learning and decision-support prototype** and should not be used as the sole basis for real lending decisions.

Important limitations include:

- The model was trained on historical credit-card customer data from Taiwan and may not represent present-day micro-lending populations.
- Predicted probabilities and decision thresholds should be validated and calibrated before operational use.
- Historical financial data may contain patterns or biases that do not generalize to other populations.
- Feature importance does not establish causation.
- A production lending system would require fairness assessment, governance, regulatory review, monitoring, security controls, and appropriate human oversight.

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Logistic Regression
- Matplotlib
- Streamlit
- Google Colab
- Git
- GitHub

## 👤 Author

**Isah Abdulmalik**

**3MTT Data Science NextGen Cohort — Capstone Project**

### Project Links

- **Live Application:** https://micro-lender-credit-risk.streamlit.app/
- **GitHub Repository:** https://github.com/maliq414/micro-lender-credit-scoring

## ⭐ Project Status

- ✅ Data preprocessing
- ✅ Exploratory data analysis
- ✅ Feature engineering
- ✅ Model training
- ✅ Model evaluation and comparison
- ✅ Credit-risk scoring framework
- ✅ Lending decision-support logic
- ✅ Streamlit application development
- ✅ GitHub version control
- ✅ Cloud deployment

**Status: Successfully Deployed 🚀**
