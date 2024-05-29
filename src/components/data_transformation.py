import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class DataTransformetaionConfig:
    data_processor_obj_file_path: str = os.path.join("artifacts", "processor.pkl")

class DataTransformetaion:
    def __init__(self):
        self.data_transformetaion_config = DataTransformetaionConfig()

    def get_transformed_data(self, df):
        try:
            ''' This function is responsible for transforming data '''
            
            # Fill missing values with mean
            df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())
            df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean())
            logging.info("Filled missing values with mean")

            # Drop irrelevant column
            df.drop(columns=["CUST_ID"], axis=1, inplace=True)
            logging.info("Dropped irrelevant columns")

            # Scale the features
            scaler = StandardScaler()
            scaled_df = scaler.fit_transform(df)
            logging.info("Scaled the features")

            return scaled_df, scaler
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_transformation(self, data_path):
        try: 
            df = pd.read_csv(data_path)
            logging.info("Data read successfully")

            logging.info("Obtaining preprocessing object")
            transformed_data, scaler = self.get_transformed_data(df)

            logging.info("Saving preprocessing object")
            save_object(file_path=self.data_transformetaion_config.data_processor_obj_file_path, obj=scaler)
            logging.info("Saved preprocessing object")

            return transformed_data, self.data_transformetaion_config.data_processor_obj_file_path
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_path = '/Users/abhi/Desktop/ML-Docker-Kubernetes/notebook/data/Customer Data.csv'
    data_transformation = DataTransformetaion()
    transformed_data, processor_path = data_transformation.initiate_transformation(data_path)

    # Assuming you have a ModelTrainer class for clustering
    # from src.components.model_trainer import ModelTrainer
    # model = ModelTrainer()
    # print(model.initiate_model_training(transformed_data))
