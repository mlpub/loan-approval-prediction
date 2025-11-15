# Loan Approval Prediction

This project builds an end-to-end machine learning model to predict loan approval.
It includes EDA, feature engineering, model training, model comparison, and deployment using FastAPI and Docker.

For full details, see the complete report:
[PROJECT-DETAIL.md](PROJECT-DETAIL.md#loan-approval-prediction) (all EDA, experiments, tuning, charts, and deployment steps)

---

## 1. Project Overview

In financial industry, loan approval is one of the most critical and sensitive processes. Banks and lending companies must carefully assess each applicant’s financial background to decide whether to approve or reject a loan request. Traditionally, this evaluation has been performed manually, which makes the process time-consuming, subjective, and prone to human error.
This project uses applicant personal information, loan information, and credit history to predict whether a loan will be paid back or default.

Main steps in this project:

* Exploratory Data Analysis (EDA)
* Data preprocessing & feature engineering
* Model training and experiments
* Parameter tuning & threshold selection
* Model evaluation (Logistic Regression & Decision Tree)
* Deployment with FastAPI
* Dockerize and deploy on cloud

The final goal is to provide a machine learning model that can assist loan officers in making better approval decisions.

---

## 2. Dataset

Source: [Loan Approval Prediction](https://www.kaggle.com/datasets/chilledwanker/loan-approval-prediction).
Total: 32,581 rows, 12 columns

The dataset contains:

* Personal information: age, income, home ownership, employment length
* Loan information: amount, intent, grade, interest rate
* Credit bureau info
* Target column `loan_status` (0 = paid, 1 = default)

---

## 3. Best Model Result

Two models were tested: Logistic Regression and Decision Tree Classifier.

The best performing model is:

### Decision Tree Classifier

* `max_depth = 8`
* `min_samples_leaf = 5`
* Best threshold = 0.62

**Test Set Performance**

| Metric    | Value |
| --------- | ----- |
| F1 Score  | 0.81  |
| Precision | 0.976 |
| Recall    | 0.693 |
| ROC AUC   | 0.897 |


---

## 4. How to Run (Local)

### Install dependencies

This project uses `uv` to manage environment and dependencies.

```
uv sync
```

### Train the model

This will generate `final_model.pkl`.

```
uv run train.py
```

### Run FastAPI (inference)

Run 
```
uv run predict.py
```

API endpoint:
`http://localhost:9696/predict`

Example input (JSON):

```
{
    "person_age": 23,
    "person_income": 115000,
    "person_home_ownership": "rent",
    "person_emp_length": 2.0,
    "loan_intent": "education",
    "loan_grade": "a",
    "loan_amnt": 35000,
    "loan_int_rate": 7.9,
    "loan_percent_income": 0.3,
    "cb_person_default_on_file": "n",
    "cb_person_cred_hist_length": 4
}
```
Response:
```
{
    'loan_probability': 0.037050231563947274, 
    'loan_status': 0
}
```


---

## 5. Docker

### Build image

```
docker build -t loan-approval-app .
```

### Run container

```
docker run -p 9600:9696 loan-approval-app
```

API available at:
`http://localhost:9600/predict`


---

## 6. Project Structure

```
loan-approval-prediction/
│ README.md                
│ PROJECT-DETAIL.md
│ train.py
│ predict.py
│ dockerfile
│ final_model.pkl
│ pyproject.toml
│ config.yaml
│ uv.lock
│ loan-approval-prediction-eda01.ipynb
│ loan-approval-prediction-model01.ipynb
│ loan-approval-prediction-model01.ipynb
├── data/
│     credit_risk_dataset.csv            
├── images/
```

---

## 7. Full Report

The full report includes:

* Full EDA
* All charts and tables
* Mutual information & correlation
* All model experiments
* Hyperparameter tuning
* Threshold optimization
* Cross validation
* Docker + Cloud deployment steps

See full report [here](PROJECT-DETAIL.md#loan-approval-prediction)

