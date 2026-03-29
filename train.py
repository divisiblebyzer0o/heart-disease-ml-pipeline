import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.classifiers import HeartdiseaseClassifier
from src.transformers.scalar import RiskScorer
from src.transformers.validator import FeatureValidator

df = pd.read_csv('data/heart.csv')
df["high_chol"] = (df["chol"] > 240).astype(int)
df["high_hr"] = (df["thalach"] > 100).astype(int)

x = df.drop(columns=["target"])
y = df["target"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

mlflow.set_experiment("heart-disease-prediction")

with mlflow.start_run():
    mlflow.log_param("n_estimators" , 50)
    mlflow.log_param("test_size" , 0.2)
    mlflow.log_param("random_state" , 42)



    pipe = Pipeline([("validator" , FeatureValidator()) , ("scalar" , RiskScorer()) , ("model" , HeartdiseaseClassifier())])

    pipe.fit(x_train , y_train)

    predictions = pipe.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    mlflow.log_metric("accuracy" , accuracy )
    mlflow.log_metric("precision" , precision)
    mlflow.log_metric("recall" , recall)

    mlflow.sklearn.log_model(pipe , "heart-disease-pipeline")


    import joblib

    joblib.dump(pipe , "heart_disease_prediction.pkl")

    print(f"accuracy: {accuracy*100:.2f}")
    print(f"precision: {precision*100:.2f}")
    print(f"recall: {recall*100:.2f}")
    print(f"run logged to mlflow")







