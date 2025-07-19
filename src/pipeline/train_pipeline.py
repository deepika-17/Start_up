import os
import sys
import pandas as pd
import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging


def main():
    try:
        logging.info("🚀 Starting training pipeline...")

        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        logging.info("✅ Data ingestion completed.")

        # 2. Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("✅ Data transformation completed.")

        # 3. Model Training
        trainer = ModelTrainer()
        model_report = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"🏁 Model training completed.\n📊 Final model report:\n{model_report}")
        print(f"\n🎯 Final Model Report:\n{model_report}")

    except Exception as e:
        logging.error("🔥 Training pipeline failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
