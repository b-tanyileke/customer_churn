# Telco Customer Churn Prediction Project

This project applies supervised machine learning techniques to predict customer churn risk for a telecom company. The main objectives are to:
- Predict churn probability (Yes/No) using logistic regression.
- Assign customers to risk levels: Low, Medium, or High.
- Analyze key factors influencing customer churn.

## Project Files

- `data\Telco_customer_churn_demographics.csv`: Contains demographic information such as age and gender.
- `data\Telco_customer_churn_location.csv`: Contains customer location details.
- `data\Telco_customer_churn_population.csv`: Contains area population data.
- `data\Telco_customer_churn_status.csv`: Contains churn labels and reasons.
- `data\telco_customer_churn_services.csv`: Contains service usage and subscription details.
- `test.ipynb`: Jupyter Notebook with all preprocessing, modeling, and analysis code.

## Project Workflow

### 1. Data Preprocessing
- Merging all datasets into a single DataFrame.
- Handling missing values.
- Encoding categorical features.

### 2. Exploratory Data Analysis (EDA)
- Understanding churn distribution.
- Identifying patterns in demographics, services, and billing.

### 3. Feature Engineering
- Creating new features.

### 4. Modeling
- Train multiple models and compare performance.
- Predicting churn probabilities.
- Classifying customers into Low, Medium, and High-risk categories.

### 5. Evaluation
- Measuring model performance with accuracy, precision, recall, and ROC-AUC.
- Analyzing false positives and false negatives for business insights.

## Getting Started

1. 
```bash
git clone https://github.com/b-tanyileke/customer_churn.git
pip install -r requirements.txt
```
2. Open `churn_prediction.ipynb` in Jupyter Notebook.
3. Run the cells to load the data, preprocess, train the model, and generate risk level predictions.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib 
- seaborn 



