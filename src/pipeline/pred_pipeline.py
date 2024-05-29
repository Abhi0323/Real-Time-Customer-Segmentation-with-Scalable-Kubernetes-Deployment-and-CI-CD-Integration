import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class Pred_Pipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            processor_path = os.path.join('artifacts', 'processor.pkl')

            model = load_object(file_path=model_path)
            transformer = load_object(file_path=processor_path)

            # Drop CUST_ID column and transform the features
            features = features.drop(columns=["CUST_ID"], axis=1)
            data_trans = transformer.transform(features)
            
            # Predict the cluster
            output = model.predict(data_trans)
            return output
        except Exception as e:
            raise CustomException(e, sys)
        
class InputData:
    def __init__(self,
                 CUST_ID: str,
                 BALANCE: float,
                 BALANCE_FREQUENCY: float,
                 PURCHASES: float,
                 ONEOFF_PURCHASES: float,
                 INSTALLMENTS_PURCHASES: float,
                 CASH_ADVANCE: float,
                 PURCHASES_FREQUENCY: float,
                 ONEOFF_PURCHASES_FREQUENCY: float,
                 PURCHASES_INSTALLMENTS_FREQUENCY: float,
                 CASH_ADVANCE_FREQUENCY: float,
                 CASH_ADVANCE_TRX: int,
                 PURCHASES_TRX: int,
                 CREDIT_LIMIT: float,
                 PAYMENTS: float,
                 MINIMUM_PAYMENTS: float,
                 PRC_FULL_PAYMENT: float,
                 TENURE: int):
        self.CUST_ID = CUST_ID
        self.BALANCE = BALANCE
        self.BALANCE_FREQUENCY = BALANCE_FREQUENCY
        self.PURCHASES = PURCHASES
        self.ONEOFF_PURCHASES = ONEOFF_PURCHASES
        self.INSTALLMENTS_PURCHASES = INSTALLMENTS_PURCHASES
        self.CASH_ADVANCE = CASH_ADVANCE
        self.PURCHASES_FREQUENCY = PURCHASES_FREQUENCY
        self.ONEOFF_PURCHASES_FREQUENCY = ONEOFF_PURCHASES_FREQUENCY
        self.PURCHASES_INSTALLMENTS_FREQUENCY = PURCHASES_INSTALLMENTS_FREQUENCY
        self.CASH_ADVANCE_FREQUENCY = CASH_ADVANCE_FREQUENCY
        self.CASH_ADVANCE_TRX = CASH_ADVANCE_TRX
        self.PURCHASES_TRX = PURCHASES_TRX
        self.CREDIT_LIMIT = CREDIT_LIMIT
        self.PAYMENTS = PAYMENTS
        self.MINIMUM_PAYMENTS = MINIMUM_PAYMENTS
        self.PRC_FULL_PAYMENT = PRC_FULL_PAYMENT
        self.TENURE = TENURE

    def transform_data_as_dataframe(self):
        try:
            user_input_data_dict = {
                "CUST_ID": [self.CUST_ID],
                "BALANCE": [self.BALANCE],
                "BALANCE_FREQUENCY": [self.BALANCE_FREQUENCY],
                "PURCHASES": [self.PURCHASES],
                "ONEOFF_PURCHASES": [self.ONEOFF_PURCHASES],
                "INSTALLMENTS_PURCHASES": [self.INSTALLMENTS_PURCHASES],
                "CASH_ADVANCE": [self.CASH_ADVANCE],
                "PURCHASES_FREQUENCY": [self.PURCHASES_FREQUENCY],
                "ONEOFF_PURCHASES_FREQUENCY": [self.ONEOFF_PURCHASES_FREQUENCY],
                "PURCHASES_INSTALLMENTS_FREQUENCY": [self.PURCHASES_INSTALLMENTS_FREQUENCY],
                "CASH_ADVANCE_FREQUENCY": [self.CASH_ADVANCE_FREQUENCY],
                "CASH_ADVANCE_TRX": [self.CASH_ADVANCE_TRX],
                "PURCHASES_TRX": [self.PURCHASES_TRX],
                "CREDIT_LIMIT": [self.CREDIT_LIMIT],
                "PAYMENTS": [self.PAYMENTS],
                "MINIMUM_PAYMENTS": [self.MINIMUM_PAYMENTS],
                "PRC_FULL_PAYMENT": [self.PRC_FULL_PAYMENT],
                "TENURE": [self.TENURE]
            }
            logging.info("Starting transformation...")
            logging.info(f"Data: {user_input_data_dict}")

            return pd.DataFrame(user_input_data_dict)
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    # Example usage:
    input_data = InputData(
        CUST_ID='C10029',
        BALANCE=7152.864372,
        BALANCE_FREQUENCY=1.0,
        PURCHASES=387.05,
        ONEOFF_PURCHASES=204.55,
        INSTALLMENTS_PURCHASES=182.5,
        CASH_ADVANCE=2236.145259,
        PURCHASES_FREQUENCY=0.666667,
        ONEOFF_PURCHASES_FREQUENCY=0.166667,
        PURCHASES_INSTALLMENTS_FREQUENCY=0.416667,
        CASH_ADVANCE_FREQUENCY=0.833333,
        CASH_ADVANCE_TRX=16,
        PURCHASES_TRX=8,
        CREDIT_LIMIT=10500.0,
        PAYMENTS=1601.448347,
        MINIMUM_PAYMENTS=1648.851345,
        PRC_FULL_PAYMENT=0.0,
        TENURE=12
    )

    pred_pipeline = Pred_Pipeline()
    input_df = input_data.transform_data_as_dataframe()
    cluster_label = pred_pipeline.predict(input_df)
    print("Cluster Label:", cluster_label)
