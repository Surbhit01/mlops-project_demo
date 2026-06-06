import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------------ #
# Helper: load processed data
# ------------------------------------------------------------------ #
def load_data(processed_dir: str):
    X_train = pd.read_csv(f"{processed_dir}/X_train.csv")
    X_test  = pd.read_csv(f"{processed_dir}/X_test.csv")
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{processed_dir}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------ #
# Helper: evaluate a trained model and return metrics dict
# ------------------------------------------------------------------ #
def evaluate(model, X_test, y_test) -> dict:
    preds      = model.predict(X_test)
    preds_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy" : round(accuracy_score(y_test, preds), 4),
        "f1_score" : round(f1_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall"   : round(recall_score(y_test, preds), 4),
        "roc_auc"  : round(roc_auc_score(y_test, preds_prob), 4),
    }


# ------------------------------------------------------------------ #
# Helper: save and log a confusion matrix as an artifact
# ------------------------------------------------------------------ #
def log_confusion_matrix(model, X_test, y_test, run_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["No Attrition", "Attrition"],
        cmap="Blues",
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {run_name}")

    os.makedirs("artifacts", exist_ok=True)
    path = f"artifacts/confusion_matrix_{run_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(path)
    print(f"  Confusion matrix saved and logged: {path}")


# ------------------------------------------------------------------ #
# RUN 1 — Logistic Regression
# ------------------------------------------------------------------ #
def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("RUN 1: Logistic Regression")
    print("=" * 50)

    # Hyperparameters
    C        = 1.0
    max_iter = 1000
    solver   = "lbfgs"

    with mlflow.start_run(run_name="logistic-regression"):

        # Logistic Regression needs scaled features to converge properly
        # We use a Pipeline: StandardScaler -> LogisticRegression
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=C, max_iter=max_iter, solver=solver))
        ])
        model.fit(X_train, y_train)

        # --- Evaluate ---
        metrics = evaluate(model, X_test, y_test)

        # --- Log parameters ---
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("solver", solver)

        # --- Log metrics ---
        print("METRICS---")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name} -> {metric_value}")
            mlflow.log_metric(metric_name, metric_value)

        # --- Log model ---
        mlflow.sklearn.log_model(model, name="model")

        # --- Log confusion matrix ---
        log_confusion_matrix(model, X_test, y_test, "logistic-regression")

        # --- Print results ---
        print(f"  Params : C={C}, max_iter={max_iter}, solver={solver}")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")

    print("Run complete.")


# ------------------------------------------------------------------ #
# RUN 2 — Random Forest (baseline)
# ------------------------------------------------------------------ #
def run_random_forest_baseline(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("RUN 2: Random Forest (baseline)")
    print("=" * 50)

    # Hyperparameters
    n_estimators = 100
    max_depth    = 5
    random_state = 42

    with mlflow.start_run(run_name="random-forest-baseline"):

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        print("METRICS---")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name} -> {metric_value}")
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, name="model")
        log_confusion_matrix(model, X_test, y_test, "random-forest-baseline")

        print(f"  Params : n_estimators={n_estimators}, max_depth={max_depth}")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")

    print("Run complete.")


# ------------------------------------------------------------------ #
# RUN 3 — XGBoost
# ------------------------------------------------------------------ #
def run_xgboost(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("RUN 3: XGBoost")
    print("=" * 50)

    # Hyperparameters
    n_estimators    = 100
    max_depth       = 4
    learning_rate   = 0.1
    random_state    = 42

    with mlflow.start_run(run_name="xgboost"):

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0
        )
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("random_state", random_state)

        print("METRICS---")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name} -> {metric_value}")
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, name="model")
        log_confusion_matrix(model, X_test, y_test, "xgboost")

        print(f"  Params : n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")

    print("Run complete.")


# ------------------------------------------------------------------ #
# RUN 4 — Random Forest (tuned)
# ------------------------------------------------------------------ #
def run_random_forest_tuned(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("RUN 4: Random Forest (tuned)")
    print("=" * 50)

    # Tuned hyperparameters
    n_estimators    = 200
    max_depth       = 10
    min_samples_split = 5
    min_samples_leaf  = 2
    random_state    = 42

    with mlflow.start_run(run_name="random-forest-tuned"):

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("random_state", random_state)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, name="model")
        log_confusion_matrix(model, X_test, y_test, "random-forest-tuned")

        print(f"  Params : n_estimators={n_estimators}, max_depth={max_depth}, "
              f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")

    print("Run complete.")


# ------------------------------------------------------------------ #
# REGISTER BEST MODEL
# ------------------------------------------------------------------ #
def register_best_model(experiment_name: str, model_name: str):
    """
    Find the best run by ROC AUC score and register it
    in the MLflow Model Registry.
    """
    print("\n" + "=" * 50)
    print("REGISTERING BEST MODEL")
    print("=" * 50)

    # Search all runs in the experiment, sorted by roc_auc descending
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.roc_auc DESC"]
    )

    best_run = runs.iloc[0]
    best_run_id   = best_run["run_id"]
    best_run_name = best_run["tags.mlflow.runName"]
    best_roc_auc  = best_run["metrics.roc_auc"]
    best_accuracy = best_run["metrics.accuracy"]

    print(f"  Best run   : {best_run_name}")
    print(f"  Run ID     : {best_run_id}")
    print(f"  ROC AUC    : {best_roc_auc}")
    print(f"  Accuracy   : {best_accuracy}")

    # Register model
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"\n  Model '{model_name}' registered.")
    print(f"  Version    : {registered.version}")
    print(f"\n  Open the MLflow UI to transition this model to 'Staging' or 'Production'.")
    print(f"  Command    : mlflow ui")
    print(f"  URL        : http://localhost:5000")


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    PROCESSED_DIR   = "data/processed"
    EXPERIMENT_NAME = "attrition-prediction"
    MODEL_NAME      = "AttritionGuard"

    # --- Load data ---
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_data(PROCESSED_DIR)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    # --- Set MLflow experiment ---
    # Creates the experiment if it doesn't exist yet
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\nMLflow experiment: '{EXPERIMENT_NAME}'")
    print("All runs will be logged here.\n")

    # --- Run all four models ---
    run_logistic_regression(X_train, X_test, y_train, y_test)
    run_random_forest_baseline(X_train, X_test, y_train, y_test)
    run_xgboost(X_train, X_test, y_train, y_test)
    run_random_forest_tuned(X_train, X_test, y_train, y_test)

    # --- Register best model ---
    register_best_model(EXPERIMENT_NAME, MODEL_NAME)

    print("\n" + "=" * 50)
    print("ALL DONE")
    print("=" * 50)
    print(f"  4 runs logged to MLflow experiment: '{EXPERIMENT_NAME}'")
    print(f"  Best model registered as: '{MODEL_NAME}'")
    print(f"\n  Start the MLflow UI to explore results:")
    print(f"  $ mlflow ui")
    print(f"  Then open: http://localhost:5000")