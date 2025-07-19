import os
import sys
import dill
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error
)

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save any Python object to disk using dill
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load any Python object from disk using dill
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Train, tune, evaluate, and compare multiple ML models using GridSearchCV.
    Returns a report dictionary with test accuracies.
    """
    try:
        report = {}

        for model_name, model in models.items():
            try:
                print(f"\n🔧 Training model: {model_name}")
                hyperparams = param.get(model_name, {})

                gs = GridSearchCV(
                    model,
                    hyperparams,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=1,
                    error_score='raise',
                    refit=True
                )

                # Handle LightGBM column naming issues
                if "LGBM" in model_name:
                    if not isinstance(X_train, pd.DataFrame):
                        X_train = pd.DataFrame(X_train, columns=[f"f_{i}" for i in range(X_train.shape[1])])
                    if not isinstance(X_test, pd.DataFrame):
                        X_test = pd.DataFrame(X_test, columns=X_train.columns)

                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_

                # Confirm fitted model
                _ = best_model.predict(X_train[:3])

                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Evaluation metrics
                train_metrics = {
                    "accuracy": accuracy_score(y_train, y_train_pred),
                    "precision": precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
                    "f1": f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
                    "mse": mean_squared_error(y_train, y_train_pred),
                    "conf_matrix": confusion_matrix(y_train, y_train_pred)
                }

                test_metrics = {
                    "accuracy": accuracy_score(y_test, y_test_pred),
                    "precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
                    "f1": f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
                    "mse": mean_squared_error(y_test, y_test_pred),
                    "conf_matrix": confusion_matrix(y_test, y_test_pred)
                }

                # Print summaries
                print(f"\n📊 {model_name} - Train Metrics:")
                for k, v in train_metrics.items():
                    if k != "conf_matrix":
                        print(f"{k.capitalize()}: {v:.4f}")
                print("Confusion Matrix:\n", train_metrics["conf_matrix"])

                print(f"\n📊 {model_name} - Test Metrics:")
                for k, v in test_metrics.items():
                    if k != "conf_matrix":
                        print(f"{k.capitalize()}: {v:.4f}")
                print("Confusion Matrix:\n", test_metrics["conf_matrix"])

                # Save accuracy to report
                report[model_name] = test_metrics["accuracy"]

            except Exception as model_err:
                print(f"⚠️ Skipping model '{model_name}' due to error:\n{model_err}")
                continue

        return report

    except Exception as e:
        raise CustomException(e, sys)
