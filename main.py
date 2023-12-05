import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import Response

# Load the logistic regression model and label encoder
loaded_pipeline = joblib.load('./Model/logistic_regression_model.joblib')
loaded_label_encoder = joblib.load('./Model/label_encoder.joblib')

# FastAPI app instance
app = FastAPI()

# Pydantic model for request input
class InputData(BaseModel):
    PRG: int 
    PL: int   
    PR: int   
    SK: int   
    TS: int   
    M11: float  
    BD2: float  
    Age: int   
    Insurance: int  

@app.get("/")
def home():
    return Response('message": "Welcome to the Sepsis Analysis API')

# Endpoint to make predictions
@app.post('/predict')
def predict(data: InputData):
    try:
        # Convert input data to a DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Preprocess the input data using the loaded pipeline
        prediction_proba = loaded_pipeline.predict_proba(input_df)
        prediction = loaded_pipeline.predict(input_df)
        prediction_encoded = loaded_label_encoder.inverse_transform([prediction])[0]

        # Extract the probability of the positive class (assuming binary classification)
        probability_positive = prediction_proba[0][1] if prediction_proba.shape[1] == 2 else None

        return {'prediction': prediction_encoded, 'probability_positive': probability_positive}
           
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
