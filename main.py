from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from src.utils import load_object
from src.exception import CustomException
from src.logger import logging

# Define the data model for the input data
class InputData(BaseModel):
    CUST_ID: str
    BALANCE: float
    BALANCE_FREQUENCY: float
    PURCHASES: float
    ONEOFF_PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    PURCHASES_FREQUENCY: float
    ONEOFF_PURCHASES_FREQUENCY: float
    PURCHASES_INSTALLMENTS_FREQUENCY: float
    CASH_ADVANCE_FREQUENCY: float
    CASH_ADVANCE_TRX: int
    PURCHASES_TRX: int
    CREDIT_LIMIT: float
    PAYMENTS: float
    MINIMUM_PAYMENTS: float
    PRC_FULL_PAYMENT: float
    TENURE: int

# Initialize the FastAPI app
app = FastAPI()

# Load the model and transformer once at startup
model_path = os.path.join('artifacts', 'model.pkl')
processor_path = os.path.join('artifacts', 'processor.pkl')

try:
    model = load_object(file_path=model_path)
    transformer = load_object(file_path=processor_path)
except Exception as e:
    logging.error(f"Error loading model or transformer: {e}")
    raise

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    try:
        # Convert input data to a DataFrame
        input_dict = input_data.model_dump(exclude_none=True)
        input_df = pd.DataFrame([input_dict])
        
        # Drop CUST_ID column
        input_df = input_df.drop(columns=["CUST_ID"], axis=1)
        
        # Transform the input data
        data_trans = transformer.transform(input_df)
        
        # Predict the cluster
        cluster_label = model.predict(data_trans)
        
        return {"cluster_label": int(cluster_label[0])}
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
