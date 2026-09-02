# House Price Prediction — Ames Housing

## 📌 Project Overview

This project builds a machine learning regression model to predict residential house sale prices using the Ames Housing dataset.

The project follows an end-to-end machine learning workflow, starting from data understanding and exploratory data analysis (EDA), followed by missing-value handling, categorical encoding, feature scaling, model training, evaluation, cross-validation, hyperparameter tuning, and final prediction generation.

The main objective was not only to build a predictive model, but also to understand **why each machine learning step is required and how different modeling choices affect performance**.

---

## 🎯 Problem Statement

Given information about a residential property — such as its quality, living area, year built, garage, basement, neighborhood, and other characteristics — predict its **SalePrice**.

This is a **supervised learning regression problem** because:

* The historical dataset contains known house sale prices.
* `SalePrice` is the target variable.
* The model learns the relationship between house features and historical sale prices.
* The trained model is then used to predict prices for previously unseen houses.

---

## 📊 Dataset

The project uses the **Ames Housing / Kaggle House Prices** dataset.

### Dataset size

* Training data: **1,460 houses**
* Test data: **1,459 houses**
* Target variable: `SalePrice`
* Original training features: **80 features + target**

The dataset contains both numerical and categorical variables describing different aspects of each property.

Examples include:

* `OverallQual`
* `GrLivArea`
* `YearBuilt`
* `TotalBsmtSF`
* `GarageCars`
* `Neighborhood`
* `KitchenQual`
* `BsmtQual`

---

## 🔎 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

* Feature distributions
* Missing values
* Categorical variables
* Feature cardinality
* Relationships between features and `SalePrice`
* Potential outliers
* Important predictive features

### Important correlations with `SalePrice`

| Feature      | Correlation |
| ------------ | ----------: |
| OverallQual  |       0.791 |
| GrLivArea    |       0.709 |
| GarageCars   |       0.640 |
| GarageArea   |       0.623 |
| TotalBsmtSF  |       0.613 |
| 1stFlrSF     |       0.606 |
| FullBath     |       0.560 |
| TotRmsAbvGrd |       0.533 |
| YearBuilt    |       0.523 |

`OverallQual` showed the strongest correlation with `SalePrice` among the examined features.

---

## 🧹 Data Preprocessing

The dataset contains both numerical and categorical features, so separate preprocessing strategies were used.

### Numerical features

Missing numerical values were handled using **median imputation**.

Median was selected because it is less sensitive to extreme values than the mean.

Numerical features were then standardized using `StandardScaler`.

The transformation is conceptually:

```text
z = (x - mean) / standard_deviation
```

This puts numerical features on a comparable scale.

---

### Categorical features

Categorical missing values were handled using:

```text
SimpleImputer(strategy="constant", fill_value="None")
```

For this dataset, many missing categorical values represent the **absence of a feature**, such as:

* No basement
* No garage
* No pool
* No fireplace

Therefore, replacing missing categorical values with `"None"` preserves the meaning of "feature absent."

Categorical features were then transformed using **OneHotEncoder**.

---

## 🔀 Train / Validation Split

The labeled training dataset was divided into:

* **80% training data:** 1,168 rows
* **20% validation data:** 292 rows

The validation set was kept separate from model training so that the model could be evaluated on previously unseen data.

---

## ⚙️ Preprocessing Pipeline

Scikit-learn `Pipeline` and `ColumnTransformer` were used to create a reproducible preprocessing workflow.

The numerical pipeline performs:

```text
Missing-value imputation
        ↓
StandardScaler
```

The categorical pipeline performs:

```text
Missing-value imputation
        ↓
OneHotEncoder
```

Both pipelines are combined using `ColumnTransformer`.

After preprocessing, the feature matrix contained **326 features**:

* 34 numerical features
* 292 one-hot encoded categorical features

---

## 🤖 Models

Two regression models were evaluated.

### 1. Linear Regression

Linear Regression was used as the initial baseline model.

Validation performance:

| Metric |     Score |
| ------ | --------: |
| MAE    | 20,781.83 |
| RMSE   | 31,740.25 |
| R²     |    0.8687 |

Training R²:

```text
0.9062
```

Validation R²:

```text
0.8687
```

The difference between training and validation performance suggested some overfitting, although the model still performed reasonably well.

---

### 2. Ridge Regression

Ridge Regression was then evaluated.

Ridge is a linear regression model with **L2 regularization**.

The regularization term discourages excessively large coefficients and can be particularly useful when the dataset contains many correlated or one-hot encoded features.

Initial Ridge validation performance:

| Metric |     Score |
| ------ | --------: |
| MAE    | 19,255.48 |
| RMSE   | 30,803.31 |
| R²     |    0.8763 |

Ridge improved the validation performance compared with the initial Linear Regression model.

---

## 🔄 Cross-Validation

Five-fold cross-validation was used during model selection.

The training data was divided into five folds.

During each iteration:

* Four folds were used for training.
* One fold was used for validation.
* Every observation was used for validation exactly once.

Because there were 1,168 training observations, each validation fold contained approximately 233–234 observations.

---

## 🎛️ Hyperparameter Tuning

The main Ridge hyperparameter tuned was `alpha`.

`alpha` controls the strength of L2 regularization.

`GridSearchCV` was used to evaluate multiple values:

```text
0.1
1
10
50
100
500
1000
```

The best value identified through cross-validation was:

```text
Best alpha = 50
```

The cross-validation results showed that `alpha = 50` produced the lowest average validation MAE among the tested values.

---

## 🏆 Final Validation Model

After hyperparameter tuning, Ridge Regression with:

```text
alpha = 50
```

was evaluated on the held-out validation set.

### Final validation performance

| Metric |     Score |
| ------ | --------: |
| MAE    | 18,624.07 |
| RMSE   | 31,530.70 |
| R²     |    0.8704 |

The Ridge model provided a more controlled coefficient structure through regularization while maintaining strong predictive performance.

---

## 🚀 Final Model

After completing model selection and hyperparameter tuning, the final Ridge model with:

```text
alpha = 50
```

was retrained using the **complete labeled training dataset**.

The final model was then used to generate predictions for the 1,459 houses in the test dataset.

### Test predictions

* Number of predictions: **1,459**
* Minimum predicted price: approximately **7,094**
* Maximum predicted price: approximately **580,757**
* Mean predicted price: approximately **178,904**
* Median predicted price: approximately **162,308**

The predictions were saved in:

```text
submissions/submission.csv
```

---

## 📁 Project Structure

```text
01-house-price-prediction/
│
├── data/
│   ├── data_description.txt
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── src/
│
└── submissions/
    └── submission.csv
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook
* VS Code
* Git
* GitHub

### Machine Learning techniques

* Exploratory Data Analysis
* Missing-value imputation
* Feature preprocessing
* One-hot encoding
* Feature scaling
* Linear Regression
* Ridge Regression
* Cross-validation
* GridSearchCV
* Model evaluation
* Regression metrics

---

## 📈 Key Learnings

Through this project, I learned how to build a complete machine learning workflow rather than training a model in isolation.

Key concepts practiced include:

1. Understanding the problem before modeling.
2. Separating features and target variables.
3. Identifying numerical and categorical features.
4. Handling missing values appropriately.
5. Understanding when missing values represent actual absence.
6. Understanding categorical feature cardinality.
7. Using `Pipeline` for reproducible preprocessing.
8. Using `ColumnTransformer` for different feature types.
9. Understanding one-hot encoding.
10. Understanding feature scaling.
11. Building a baseline Linear Regression model.
12. Understanding overfitting using training vs validation performance.
13. Understanding regularization through Ridge Regression.
14. Using cross-validation.
15. Performing hyperparameter tuning with `GridSearchCV`.
16. Evaluating models using MAE, RMSE and R².
17. Training a final model using all available labeled data.
18. Generating predictions for unseen test data.
19. Structuring an ML project for GitHub and portfolio use.

---

## 🔮 Future Improvements

Possible improvements for future versions include:

* Log-transforming the `SalePrice` target.
* Testing Elastic Net and Lasso regression.
* Comparing tree-based models such as Random Forest and Gradient Boosting.
* Testing advanced boosting algorithms.
* Performing more systematic feature engineering.
* Investigating and handling influential outliers.
* Building a reusable training script inside `src/`.
* Adding automated model evaluation.
* Creating a prediction API.
* Deploying the model as a small web application.

---

## 👤 Project Purpose

This project is part of my personal Machine Learning portfolio.

The goal is to demonstrate practical understanding of the complete machine learning lifecycle, including data analysis, preprocessing, modeling, evaluation, experimentation, and reproducible project organization.
