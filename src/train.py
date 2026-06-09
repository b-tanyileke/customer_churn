"""
This script trains multiple classification models to predict customer churn using the preprocessed data. 
It logs parameters, metrics, and artifacts (like confusion matrices and feature importance plots) to MLflow for experiment tracking. 
Finally, it registers the best performing model based on F1 score in the MLflow Model Registry.

"""
import os
import tempfile
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocess import prepare_data
from config import RANDOM_STATE
from explain import explain_model_sample
from explain import get_feature_names

warnings.filterwarnings("ignore")


def build_pipeline(model, categorical_columns, numerical_columns):
    """Builds one pipeline that handles raw features and model training."""
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns
            ),
            (
                "numerical",
                PowerTransformer(),
                numerical_columns
            )
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("classifier", model)
        ]
    )


# MLFLOW CONFIG
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("telco-churn-prediction")


# LOAD DATA
X_train, X_test, y_train, y_test, categorical_columns, numerical_columns = prepare_data()


# debug test
print(f"Train Shape: {X_train.shape}, {y_train.shape}")
print(f"Test Shape: {X_test.shape}, {y_test.shape}")
print(f"Train Target Counts:\n{y_train.value_counts()}")
print(f"Test Target Counts:\n{y_test.value_counts()}")


# MODEL DEFINITIONS
models = {

    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance"
    ),

    "LogisticRegression": LogisticRegression(
        class_weight="balanced",
        C=0.1,
        solver="liblinear",
        random_state=RANDOM_STATE
    ),

    "DecisionTree": DecisionTreeClassifier(
        criterion="entropy",
        max_depth=7,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=20,
        criterion="entropy",
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=20,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
}


# TRAIN MODELS

results = []

best_model = None
best_model_name = None
best_f1 = 0


for model_name, model in models.items():

    print(f"\nTraining {model_name}")

    with mlflow.start_run(run_name=model_name):

        pipeline = build_pipeline(model, categorical_columns, numerical_columns)

        # LOG PARAMETERS
        params = model.get_params()

        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass

        # TRAIN
        pipeline.fit(X_train, y_train)

        # PREDICT
        y_pred = pipeline.predict(X_test)

        # probability scores if available
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)

        else:
            roc_auc = roc_auc_score(y_test, y_pred)

        # METRICS
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc
        }

        mlflow.log_metrics(metrics)


        # CONFUSION MATRIX
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 5))

        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)

        plt.title(model_name)

        temp_dir = tempfile.mkdtemp()

        cm_path = os.path.join(temp_dir,f"{model_name}_confusion_matrix.png")

        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        # FEATURE IMPORTANCE
        trained_model = pipeline.named_steps["classifier"]

        if hasattr(trained_model, "feature_importances_"):

            importance_df = pd.DataFrame(
                {
                    "Feature": get_feature_names(pipeline),
                    "Importance": trained_model.feature_importances_
                }
            )
            importance_df = importance_df.sort_values("Importance",ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            importance_df.head(15).plot(kind="bar",x="Feature",y="Importance",ax=ax)
            plt.title(f"{model_name} Feature Importance")
            plt.xticks(rotation=45, ha="right")

            importance_path = os.path.join(temp_dir,f"{model_name}_feature_importance.png")

            plt.savefig(importance_path)
            plt.close()

            mlflow.log_artifact(importance_path)

        # LOG MODEL
        mlflow.sklearn.log_model(sk_model=pipeline,name=model_name)

        # TRACK BEST MODEL
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_model_name = model_name

        results.append({"Model": model_name, **metrics})

    # debug message
    print(f"{model_name} F1: {f1:.4f}")
    print(f"{model_name} ROC-AUC: {roc_auc:.4f}")


# RESULTS TABLE
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("f1_score",ascending=False)

print("\nModel Comparison")
print(results_df)

results_df.to_csv("model_comparison.csv",index=False)

# REGISTER BEST MODEL
print(f"\nBest Model: {best_model_name}")

with mlflow.start_run(run_name=f"{best_model_name}_REGISTERED"):
    top_shap_features, summary_path, bar_path = explain_model_sample(
        best_model,
        best_model_name,
        X_test
    )

    mlflow.log_artifact(summary_path)
    mlflow.log_artifact(bar_path)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="best_model",
        registered_model_name="telco_churn_model"
    )
    mlflow.log_metric("best_f1",best_f1)

print("\nTop 10 SHAP Features")
print(top_shap_features)
print("\nTraining Complete")
