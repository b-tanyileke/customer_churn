import mlflow

from mlflow.tracking import MlflowClient
from preprocess import load_data
from config import RANDOM_STATE
from explain import build_shap_explainer
from explain import explain_single_prediction
from explain import get_feature_names


mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "telco_churn_model"
MODEL_URI = f"models:/{MODEL_NAME}/latest"

# Load the best registered pipeline from MLflow.
model = mlflow.pyfunc.load_model(MODEL_URI)
model_version = "latest"
_explainer = None
_feature_names = None


def get_model_version():
    """Gets the latest registered model version from MLflow if available."""
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")

        if versions:
            latest_version = max(versions, key=lambda item: int(item.version))
            return latest_version.version

    except Exception:
        pass

    return "latest"


model_version = get_model_version()


def risk_level(prob):
    """Categorizes the risk level based on the predicted probability of churn.
    Args:
        prob (float): The predicted probability of churn.
    Returns:
        str: The risk level category ("Low Risk", "Medium Risk", "High Risk").
    """
    if prob <= 0.3:
        return "Low Risk"

    elif prob <= 0.7:
        return "Medium Risk"

    return "High Risk"


def get_shap_explainer():
    """Builds a SHAP explainer for the loaded MLflow pipeline."""
    global _explainer
    global _feature_names

    if _explainer is not None:
        return _explainer, _feature_names

    sklearn_model = model._model_impl.sklearn_model

    background_data = load_data().drop(columns=["Churn Value"])
    background_sample = background_data.sample(
        n=min(50, len(background_data)),
        random_state=RANDOM_STATE
    )

    _feature_names = get_feature_names(sklearn_model)
    _explainer = build_shap_explainer(sklearn_model, background_sample)

    return _explainer, _feature_names


def get_top_factors(features):
    """Returns the top 3 features affecting one churn prediction."""
    sklearn_model = model._model_impl.sklearn_model
    explainer, feature_names = get_shap_explainer()

    return explain_single_prediction(
        sklearn_model,
        features,
        explainer,
        feature_names,
        top_n=3
    )


def predict_customer(features):
    """Predicts churn probability and risk level from raw customer features.

    Args:
        features (pd.DataFrame): A DataFrame containing the original raw feature columns.

    Returns:
        dict: A dictionary containing the predicted probability of churn and the corresponding risk level.
    """

    sklearn_model = model._model_impl.sklearn_model
    prediction = sklearn_model.predict(features)[0]
    prob = sklearn_model.predict_proba(features)[0][1]
    prediction_label = "Churn" if prediction == 1 else "No Churn"

    return {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "prediction": prediction_label,
        "probability": float(prob),
        "risk_level": risk_level(prob),
        "top_factors": get_top_factors(features)
    }
