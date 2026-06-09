"""Load and split the customer churn data.

The actual preprocessing is done inside the scikit-learn Pipeline in train.py.
This keeps training and prediction preprocessing identical.
"""

import pandas as pd

from sklearn.model_selection import train_test_split

from config import *


def load_data():
    """Loads the customer churn data from mutiple CSV files 
    and merges them into a single DataFrame
    Returns:
        pd.DataFrame: A DataFrame containing the merged customer churn data.
    """
    demographics = pd.read_csv(DEMOGRAPHICS)
    services = pd.read_csv(SERVICES)
    status = pd.read_csv(STATUS)

    training_data = demographics[["Dependents","Married","Senior Citizen"]].copy()

    training_data[["Satisfaction Score","CLTV"]] = status[["Satisfaction Score","CLTV"]]

    training_data = training_data.join(
        services[
            [
                "Internet Service",
                "Online Security",
                "Premium Tech Support",
                "Contract",
                "Paperless Billing",
                "Monthly Charge",
                "Tenure in Months"
            ]
        ]
    )

    training_data["Churn Value"] = status["Churn Value"]

    return training_data


def prepare_data():
    """Prepares raw customer churn data for the training pipeline.

    Returns:
        tuple: Training features, testing features, training labels, testing labels,
        categorical columns, and numerical columns.
    """
    df = load_data()

    y = df["Churn Value"]

    X = df.drop(columns=["Churn Value"])

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test, categorical_columns, numerical_columns
