import joblib
import numpy as np
import os

from src.models.classifiers import HeartdiseaseClassifier
from src.transformers.scalar import RiskScorer
from src.transformers.validator import FeatureValidator

def safe_predict(pipeline , x):
    try: 
        predictions = pipeline.predict(x)
        return predictions
    except Exception as e:
        print(f"prediction failed: {e}")
        return None
    
current_dir = os.path.dirname(__file__)

model_path = os.path.join(current_dir,"artifacts" , "heart_disease_prediction.pkl")

pipeline = joblib.load(model_path)

np.random.seed(42)
x_test = np.random.randn(50,5)

predictions = safe_predict(pipeline , x_test)

print(predictions)


