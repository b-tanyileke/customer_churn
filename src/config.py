"""This module contains configuration settings for the customer churn analysis, 
including file paths for the data and parameters for data splitting and random state.
"""


from pathlib import Path

DATA_DIR = Path("data")

DEMOGRAPHICS = DATA_DIR / "Telco_customer_churn_demographics.csv"
LOCATION = DATA_DIR / "Telco_customer_churn_location.csv"
SERVICES = DATA_DIR / "Telco_customer_churn_services.csv"
STATUS = DATA_DIR / "Telco_customer_churn_status.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
