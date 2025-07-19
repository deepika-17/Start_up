import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, numerical_features, categorical_features):
        try:
            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Combine pipelines
            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_features),
                ("cat", cat_pipeline, categorical_features)
            ])

            logging.info("Preprocessor pipeline created successfully.")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded.")

            # ✅ Define selected features (used in model and form)
            selected_features = [
                'funding_total_usd', 'funding_rounds',
                'is_CA', 'is_NY', 'is_MA', 'is_TX',
                'is_software', 'is_web', 'is_mobile', 'is_biotech',
                'age_first_funding_year', 'age_last_funding_year',
                'state_code', 'category_code', 'status'  # include target
            ]

            # ✅ Keep only selected features
            train_df = train_df[selected_features]
            test_df = test_df[selected_features]

            target_column = "status"

            # Identify feature types
            numerical_features = ['funding_total_usd', 'funding_rounds',
                                  'is_CA', 'is_NY', 'is_MA', 'is_TX',
                                  'is_software', 'is_web', 'is_mobile', 'is_biotech',
                                  'age_first_funding_year', 'age_last_funding_year']
            categorical_features = ['state_code', 'category_code']

            # Split features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]
            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Get preprocessor
            preprocessing_obj = self.get_data_transformer_object(numerical_features, categorical_features)

            # Transform
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            # Combine transformed features with target
            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Data transformation completed and preprocessor saved.")
            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
