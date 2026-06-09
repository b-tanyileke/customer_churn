"""Small SHAP helper functions for training and prediction."""

import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import shap

from config import RANDOM_STATE


def get_feature_names(pipeline):
    """Gets transformed feature names from a fitted pipeline."""
    preprocessing = pipeline.named_steps["preprocessing"]
    return preprocessing.get_feature_names_out()


def transform_features(pipeline, features):
    """Applies the fitted preprocessing step and returns a DataFrame."""
    preprocessing = pipeline.named_steps["preprocessing"]
    feature_names = get_feature_names(pipeline)

    return pd.DataFrame(
        preprocessing.transform(features),
        columns=feature_names
    )


def build_shap_explainer(pipeline, background_data):
    """Builds a SHAP explainer using preprocessed background data."""
    classifier = pipeline.named_steps["classifier"]
    transformed_background = transform_features(pipeline, background_data)

    return shap.Explainer(
        classifier.predict_proba,
        transformed_background.to_numpy(),
        feature_names=get_feature_names(pipeline)
    )


def get_churn_shap_values(shap_values):
    """Selects SHAP values for the churn class when class values are present."""
    if len(shap_values.values.shape) == 3:
        return shap_values[:, :, 1]

    return shap_values


def get_shap_importance(shap_values, feature_names):
    """Calculates feature importance from mean absolute SHAP values."""
    # Mean absolute SHAP values are used as feature importance:
    # larger values mean the feature had a bigger average impact on predictions.
    mean_abs_shap = abs(shap_values.values).mean(axis=0)

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Mean Absolute SHAP Value": mean_abs_shap
        }
    )

    return importance.sort_values("Mean Absolute SHAP Value", ascending=False)


def explain_model_sample(pipeline, model_name, X_test):
    """Creates SHAP plots and returns top global features for the best model."""
    shap_sample = X_test.sample(
        n=min(100, len(X_test)),
        random_state=RANDOM_STATE
    )

    background_sample = X_test.sample(
        n=min(50, len(X_test)),
        random_state=RANDOM_STATE
    )

    feature_names = get_feature_names(pipeline)
    transformed_sample = transform_features(pipeline, shap_sample)

    # SHAP values show how much each feature pushes a prediction up or down.
    # Here we explain the probability of churn, which is class 1.
    explainer = build_shap_explainer(pipeline, background_sample)
    shap_values = explainer(transformed_sample.to_numpy(), silent=True)
    churn_shap_values = get_churn_shap_values(shap_values)
    shap_importance = get_shap_importance(churn_shap_values, feature_names)

    temp_dir = tempfile.mkdtemp()
    summary_path = os.path.join(temp_dir, f"{model_name}_shap_summary.png")
    bar_path = os.path.join(temp_dir, f"{model_name}_shap_bar.png")

    shap.summary_plot(
        churn_shap_values,
        transformed_sample,
        show=False
    )
    plt.title(f"{model_name} SHAP Summary")
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()

    shap.plots.bar(
        churn_shap_values,
        max_display=15,
        show=False
    )
    plt.title(f"{model_name} SHAP Feature Importance")
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()

    return shap_importance.head(10), summary_path, bar_path


def explain_single_prediction(pipeline, features, explainer, feature_names, top_n=3):
    """Returns the top local SHAP factors for one customer prediction."""
    transformed_features = transform_features(pipeline, features)
    shap_values = explainer(transformed_features.to_numpy(), silent=True)

    if len(shap_values.values.shape) == 3:
        churn_values = shap_values.values[0, :, 1]
    else:
        churn_values = shap_values.values[0]

    # Positive values push the customer closer to churn, negative values push away.
    top_indexes = abs(churn_values).argsort()[-top_n:][::-1]

    return [
        {
            "feature": str(feature_names[index]),
            "impact": float(churn_values[index])
        }
        for index in top_indexes
    ]
