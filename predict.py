import joblib
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))

from src.models.classifiers import HeartdiseaseClassifier
from src.transformers.scalar import RiskScorer
from src.transformers.validator import FeatureValidator

def safe_predict(pipeline , x):
    try:
        predictions = pipeline.predict(x)
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
    

model_path = os.path.join(os.path.dirname(__file__),"artifacts", "heart_disease_prediction.pkl")
pipeline = joblib.load(model_path)
print("pipeline loaded")

df = pd.read_csv("data/heart.csv")
df["high_chol"] = (df["chol"] > 240).astype(int)
df["high_hr"] = (df["thalach"] > df["thalach"].mean()).astype(int)

x = df.drop(columns=["target"])
y = df["target"].values


predictions = safe_predict(pipeline , x)

for i , (pred , actual) in enumerate(zip(predictions[:10] , y[:10])):
    risk = "High" if pred == 0 else "LOW"
    print(f"patient {i+1}: predicted = {risk} , actual = {'sick' if actual == 0 else 'healthy'}")


