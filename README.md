# Health Insurance Claim Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting **health insurance claim amounts** using Machine Learning techniques. The project includes data cleaning, exploratory data analysis (EDA), categorical encoding, model training, evaluation, feature importance analysis, and prediction for new customers.

The main objective is to understand the factors that influence insurance claim amounts and build a machine learning model capable of predicting the expected claim for a customer.

---

## 🎯 Objectives

* Perform data cleaning and preprocessing.
* Explore the dataset using EDA.
* Analyze relationships between customer attributes and insurance claims.
* Convert categorical features into numerical form.
* Train and compare Machine Learning models.
* Evaluate model performance using regression metrics.
* Identify important features affecting insurance claims.
* Save the trained model for future predictions.
* Predict the claim amount for a new customer.

---

## 📊 Dataset

The dataset contains **15,000 records** and the following features:

| Feature               | Description                              |
| --------------------- | ---------------------------------------- |
| `age`                 | Age of the customer                      |
| `sex`                 | Gender of the customer                   |
| `weight`              | Customer weight                          |
| `bmi`                 | Body Mass Index                          |
| `hereditary_diseases` | Hereditary disease information           |
| `no_of_dependents`    | Number of dependents                     |
| `smoker`              | Smoking status                           |
| `city`                | Customer's city                          |
| `bloodpressure`       | Blood pressure                           |
| `diabetes`            | Diabetes status                          |
| `regular_ex`          | Regular exercise status                  |
| `job_title`           | Customer's occupation                    |
| `claim`               | Insurance claim amount (target variable) |

---

## 🔍 Exploratory Data Analysis

The following analysis was performed:

* Univariate Analysis
* Bivariate Analysis
* Distribution analysis
* Mean and median comparison
* Skewness analysis
* Categorical feature analysis
* Relationship between customer characteristics and claim amounts

Some important observations included:

* Smoking status showed a strong relationship with claim amount.
* Age and BMI also contributed significantly to claim prediction.
* Claim amounts showed a right-skewed distribution.
* Different customer characteristics produced noticeable differences in average claim amounts.

---

## 🛠️ Data Preprocessing

The following preprocessing steps were performed:

1. Missing values were handled.
2. Numeric columns were converted to appropriate numeric formats.
3. Categorical variables were identified.
4. Categorical variables were converted using **One-Hot Encoding**.
5. Features and target variable were separated.
6. Dataset was divided into training and testing sets.

After preprocessing and encoding:

```text
Features: 142
Training samples: 11,123
Testing samples: 2,781
```

---

## 🤖 Machine Learning Models

Two regression approaches were used during the project:

### 1. Linear Regression

Linear Regression was used as a baseline model to establish an initial performance reference.

### 2. Random Forest Regressor

Random Forest Regressor was used as the main model because it can capture non-linear relationships between different customer characteristics and insurance claims.

The Random Forest model achieved strong predictive performance on the test dataset.

---

## 📈 Model Evaluation

The final Random Forest model achieved:

| Metric      |       Result |
| ----------- | -----------: |
| MAE         |       524.32 |
| MSE         | 4,177,827.51 |
| RMSE        |     2,043.97 |
| R² Score    |      0.97208 |
| Training R² |      0.99485 |

### Interpretation

The **R² score of 0.97208** indicates that the model explains approximately **97.2% of the variation** in the test-set claim amounts.

The MAE of approximately **524.32** means that, on average, the model's prediction differs from the actual claim by around 524 units of the target variable.

---

## ⭐ Feature Importance

The Random Forest model identified the following important features:

| Feature                         | Importance |
| ------------------------------- | ---------: |
| `smoker`                        |   0.596164 |
| `age`                           |   0.115636 |
| `bmi`                           |   0.111279 |
| `hereditary_diseases_NoDisease` |   0.026453 |
| `weight`                        |   0.021016 |
| `bloodpressure`                 |   0.020549 |
| `diabetes`                      |   0.020119 |
| `no_of_dependents`              |   0.013383 |

The feature importance results show that **smoking status had the largest contribution to the Random Forest model's predictions**, followed by age and BMI.

---

## 💰 New Customer Prediction

The trained model was also tested with a new customer profile.

Example customer:

```text
Age              : 35
Weight           : 70
BMI              : 24.5
Dependents       : 2
Smoker           : No
Blood Pressure   : 80
Diabetes         : No
Regular Exercise : Yes
Sex              : Male
City             : Delhi
Job              : Engineer
```

The model predicted an estimated claim of approximately:

```text
₹6,721.56
```

---

## 💾 Model Saving

The trained Random Forest model was saved using Joblib:

```python
import joblib

joblib.dump(model, 'health_insurance_model.pkl')
```

The saved model can later be loaded without retraining:

```python
loaded_model = joblib.load('health_insurance_model.pkl')
```

---

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
EDA
     ↓
Univariate Analysis
     ↓
Bivariate Analysis
     ↓
Categorical Encoding
     ↓
Train-Test Split
     ↓
Linear Regression
     ↓
Random Forest Regression
     ↓
Model Evaluation
     ↓
Feature Importance
     ↓
Model Saving
     ↓
New Customer Prediction
```

---

## 🧰 Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

## 📁 Project Files

```text
health-insurance-claim-prediction/
│
├── health_insurance_claim_prediction.ipynb
├── health_insurance_model.pkl
└── README.md
```

---

## 🚀 Future Improvements

Possible future improvements include:

* Hyperparameter tuning
* Cross-validation
* Additional feature engineering
* Further outlier analysis
* Trying additional regression algorithms
* Model deployment using Flask or FastAPI
* Creating a web interface for real-time prediction

---

## 📌 Conclusion

This project demonstrates an end-to-end Machine Learning workflow for predicting health insurance claim amounts. It covers data preprocessing, exploratory analysis, feature encoding, model training, evaluation, feature importance, model persistence, and prediction on new customer data.

The final Random Forest model achieved an **R² score of 0.97208 on the test dataset**, demonstrating strong predictive performance on this dataset.
