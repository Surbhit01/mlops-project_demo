import mlflow
import mlflow.sklearn
import pickle


# def load_from_mlflow(model_name: str, version: str):
#     """Load model from MLflow Model Registry by version number."""
    
#     mlflow.set_tracking_uri("sqlite:///mlflow.db")
#     model_uri = f"models:/{model_name}/{version}"
#     print(f"Loading model from MLflow: {model_uri}")
#     # Load as sklearn model directly — gives us predict_proba
#     return mlflow.sklearn.load_model(model_uri)


def load_from_pkl(model_path: str):
    """Load model from a local pickle file."""
    print(f"Loading model from local file: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)