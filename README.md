Customer Churn Prediction

A machine learning classification project that predicts whether a telecom customer will churn (cancel their subscription), using Logistic Regression with SMOTE to handle class imbalance.

Problem Statement

Customer churn is costly for subscription-based businesses — acquiring a new customer is typically far more expensive than retaining an existing one. This project builds a classification model to identify customers likely to churn, so the business can proactively intervene.

Dataset
Source: Telco Customer Churn (public dataset, originally released by IBM)
Size: 7,043 customers, 21 features (demographics, account info, services subscribed)
Target: Churn (Yes/No)
Class distribution: ~73% No Churn, ~27% Churn (imbalanced)
Approach
Data Cleaning
Converted TotalCharges from text to numeric (11 rows had blank-space values for new customers with 0 tenure)
Dropped customerID (identifier, no predictive value)
Encoding
Multi-category columns (Contract, PaymentMethod, InternetService, etc.) → one-hot encoded with pd.get_dummies(), to avoid implying a false numeric order between categories
Binary columns (gender, Partner, PaperlessBilling, etc.) → label encoded
Handling Class Imbalance
Applied SMOTE (Synthetic Minority Oversampling) to the training set only, to avoid data leakage into the test set
Feature Scaling
Standardized tenure, MonthlyCharges, and TotalCharges with StandardScaler (fit on train, applied to test)
Model
Logistic Regression, evaluated on a held-out 30% test set
Results
Metric	Class 0 (No Churn)	Class 1 (Churn)
Precision	0.89	0.53
Recall	0.79	0.70
F1-score	0.84	0.60

Overall accuracy: 77%

Recall on the churn class (70%) was prioritized as the key metric over raw accuracy, since missing an actual churner (false negative) is more costly to the business than a false alarm.

Key Debugging Insights

While building this project, two data-integrity bugs were caught and fixed:

Silent data corruption: TotalCharges (a continuous dollar value) was originally label-encoded before being cleaned, turning real charge amounts into meaningless category IDs. Fixed by cleaning the column before any encoding step.
Target leakage via positional indexing: using X = df.iloc[:,:-1] / y = df.iloc[:,-1] broke after one-hot encoding shifted column order, causing the model to accidentally train on the wrong target column. Fixed by selecting X/y explicitly by column name.
Tech Stack

Python, Pandas, scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Seaborn

Next Steps
Compare against Random Forest / other classifiers
Use stratify=y in the train/test split
Hyperparameter tuning
Feature importance analysis to identify top churn drivers
How to Run
bash
pip install pandas scikit-learn imbalanced-learn matplotlib seaborn

Open Customer_prediction_churn1.ipynb and run all cells. The dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv) is included in this repo.
