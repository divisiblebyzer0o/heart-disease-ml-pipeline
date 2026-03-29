from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))

from src.models.classifiers import HeartdiseaseClassifier
from src.transformers.scalar import RiskScorer
from src.transformers.validator import FeatureValidator

pipeline = joblib.load("artifacts/heart_disease_prediction.pkl")





class Patient(BaseModel):
    age: int
    sex:int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int
    high_col: int
    high_hr: int


app = FastAPI()

@app.get("/")
def home():
    return {"message" : "heart disease API is running"}

@app.post("/predict")
def predict(patient: Patient):
    
    features = np.array([[
        patient.age, patient.sex, patient.cp,
        patient.trestbps, patient.chol, patient.fbs,
        patient.restecg, patient.thalach, patient.exang,
        patient.oldpeak, patient.slope, patient.ca,
        patient.thal, patient.high_col, patient.high_hr
    ]])

    prediction = pipeline.predict(features)[0]
    probability = pipeline.predict_proba(features)[0]

    return {
        "prediction" : int(prediction),
        "risk" : "HIGH" if prediction == 0 else "LOW" ,
        "confidance" : round(float(probability[prediction]) , 3)
    }





