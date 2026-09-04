# Customer Churn Prediction

## Project Overview

Customer churn prediction is a binary classification problem where the goal is to identify customers who are likely to leave a service.

In this project, I built an end-to-end machine learning pipeline using the Telco Customer Churn dataset. The project covers data understanding, preprocessing, model training, evaluation, hyperparameter tuning, threshold analysis, and saving the final model for future predictions.

## Business Problem

For a telecom company, identifying customers who are likely to churn can help the business take preventive actions such as targeted offers, improved support, or customer retention campaigns.

The objective of this project is to predict whether a customer will churn based on their demographic information, services, contract details, tenure, and billing information.

## Dataset

The project uses the Telco Customer Churn dataset.

* Rows: 7,043
* Original columns: 21
* Target variable: `Churn`
* Target classes:

  * `No`
  * `Yes`

The dataset contains information about customer demographics, account information, subscribed services, contract type, payment method, and charges.

## Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

1. Load and understand the dataset
2. Explore the data
3. Identify missing values
4. Clean and prepare the data
5. Separate features and target
6. Remove the customer identifier
7. Split data into training and test sets
8. Encode categorical features
9. Impute missing numerical values
10. Scale numerical features
11. Train multiple classification models
12. Evaluate model performance
13. Perform hyperparameter tuning
14. Analyze classification thresholds
15. Interpret Logistic Regression coefficients
16. Build a production-style pipeline
17. Save the trained pipeline
18. Load the saved pipeline
19. Make predictions on a completely new customer

## Preprocessing

### Numerical Features

The following numerical features were used:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

Numerical preprocessing includes:

* Median imputation
* Standardization using `StandardScaler`

### Categorical Features

Categorical variables were transformed using `OneHotEncoder`.

`handle_unknown="ignore"` was used so that the pipeline can safely process previously unseen categorical values.

### Target Encoding

The target variable was converted from:

```text
No  → 0
Yes → 1
```

The `customerID` column was removed because it is an identifier rather than a predictive feature.

## Models Evaluated

Three classification algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest

Random Forest hyperparameters were optimized using `GridSearchCV` with 5-fold cross-validation and ROC-AUC as the optimization metric.

## Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   80.55% |    65.72% | 55.88% |   60.40% |  84.20% |
| Decision Tree       |   79.84% |    63.47% | 56.68% |   59.89% |  82.97% |
| Random Forest       |   79.49% |    67.49% | 43.85% |   53.16% |  84.07% |

Logistic Regression provided the strongest overall performance among the evaluated models at the default classification threshold.

## Threshold Analysis

The default classification threshold is 0.50, but changing the threshold changes the balance between precision and recall.

For the tuned Random Forest:

| Threshold | Accuracy | Precision | Recall |     F1 |
| --------: | -------: | --------: | -----: | -----: |
|      0.30 |   74.80% |    51.69% | 77.54% | 62.03% |
|      0.40 |   79.49% |    60.55% | 65.24% | 62.81% |
|      0.50 |   79.49% |    67.49% | 43.85% | 53.16% |
|      0.60 |   78.07% |    75.59% | 25.67% | 38.32% |
|      0.70 |   74.10% |    90.91% |  2.67% |  5.19% |

This demonstrates that the classification threshold can be adjusted depending on the business objective.

For example, if the business wants to identify more potential churners, a lower threshold can increase recall.

## Important Features

The Logistic Regression coefficients provided useful insights into which features were associated with higher or lower predicted churn.

Some stronger positive churn signals included:

* Fiber optic internet service
* Month-to-month contracts
* Electronic check payment
* Lack of online security
* Lack of technical support

Some stronger negative churn signals included:

* Longer customer tenure
* Two-year contracts
* DSL internet service

These coefficients describe associations learned by the model and should not be interpreted as proof of causation.

## Final Machine Learning Pipeline

The final model was packaged into a single scikit-learn `Pipeline` containing:

```text
Raw Customer Data
       ↓
ColumnTransformer
       ↓
Numerical preprocessing
       ↓
Categorical preprocessing
       ↓
Logistic Regression
       ↓
Prediction
```

The complete pipeline was saved as:

```text
models/customer_churn_pipeline.pkl
```

Because preprocessing and the model are stored together, new customer data can be passed directly to the saved pipeline without manually repeating the preprocessing steps.

## Example Prediction

A completely new customer record was passed directly to the saved pipeline.

The model predicted:

```text
Will customer churn?: Yes
Churn probability: 78.51%
```

This demonstrates how the trained model can be used on new customer information after training.

## Project Structure

```text
02-customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│
├── src/
│
├── models/
│   └── customer_churn_pipeline.pkl
│
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook
* Git
* GitHub

## Key Concepts Learned

This project helped demonstrate practical understanding of:

* Binary classification
* Class imbalance
* Train/test splitting
* Stratification
* One-hot encoding
* Feature scaling
* Missing-value handling
* Logistic Regression
* Decision Trees
* Random Forest
* Cross-validation
* GridSearchCV
* Confusion Matrix
* Precision
* Recall
* F1 Score
* ROC-AUC
* ROC curves
* Classification thresholds
* Model coefficients
* ML pipelines
* Model persistence using Joblib
* Making predictions on new data

## Conclusion

This project demonstrates an end-to-end approach to solving a real-world customer churn prediction problem, from raw data preprocessing through model evaluation and deployment-ready model serialization.

The final Logistic Regression pipeline achieved an ROC-AUC of approximately 84.20% and can be loaded later to generate predictions for new customers.
