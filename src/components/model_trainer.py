import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

warnings.filterwarnings("ignore")

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing arrays into features and target.")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Applying SMOTE for class balancing.")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(probability=True),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
                "Naive Bayes": GaussianNB()
            }

            params = {
                "Logistic Regression": {},
                "Decision Tree": {"max_depth": [3, 5, 10]},
                "Random Forest": {"n_estimators": [50, 100], "max_depth": [5, 10]},
                "KNN": {"n_neighbors": [3, 5, 7]},
                "SVM": {"C": [0.5, 1.0], "kernel": ["linear", "rbf"]},
                "Gradient Boosting": {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1]},
                "XGBoost": {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1]},
                "Naive Bayes": {}
            }

            logging.info("Evaluating models...")
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            # Select best model
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)

            # Predict and evaluate final metrics
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            try:
                auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
            except:
                auc = 0.0

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}, AUC: {auc:.3f}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return {
                "model_name": best_model_name,
                "accuracy": accuracy,
                "f1_score": f1,
                "auc_score": auc
            }

        except Exception as e:
            raise CustomException(e, sys)
