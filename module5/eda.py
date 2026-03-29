import pandas as pd
import numpy as np

import sys 
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

print(os.getcwd())
print(sys.path)

df = pd.read_csv("data/heart.csv")


print(df.shape)

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

print(df.dtypes)

print(df["target"].value_counts())

print(df.groupby("target")["age"].mean())

print(df.groupby("target")["chol"].mean())

print(df["sex"].value_counts())

print(f"max heart-rate {df['thalach'].max()}")

print(f"min heart-rate {df["thalach"].min()}")

print(df.groupby("target")["sex"].value_counts())

df["age_group"] = pd.cut(df["age"], bins = [0 , 40 , 50 , 60 , 100] , labels = ["young" , "middle" , "senior", "elderly"])

print(df["age_group"].value_counts())


df["high_col" ] = (df["chol"] > 240).astype(int)

print(df["high_col"].value_counts())

df["high_hr"] = (df["thalach"] > df["thalach"].mean()).astype(int)

print(df["high_hr"].value_counts())

df = df.drop(columns = ["age_group"])

x = df.drop(columns = ["target"]).values

y = df["target"].values

print(df.groupby("target")["thalach"].mean())
print(df.groupby("target")["age"].mean())

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

print(f"x_train shape {x_train.shape}")
print(f"x_test shape {x_test.shape}")
print(f"y_train shape {y_train.shape}")
print(f"y_test shape {y_test.shape}")


from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score , classification_report

from src.models.classifiers import HeartdiseaseClassifier
from src.transformers.scalar import RiskScorer
from src.transformers.validator import FeatureValidator
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score

# mlflow.set_tracking_uri("file:///c:/Other/ML road/Python demo/module4/heart_disease_project/mlruns")

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

# prdictions = pipe.predict(x_test)

# accuracy = accuracy_score(y_test , prdictions)

# print(accuracy) 
# print(classification_report(y_test , prdictions))

# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(pipe , x,y , cv=5 , scoring="accuracy")
# print(f"cv score: {scores}")
# print(f"Mean: {scores.mean()*100:.2f}%")
# print(f"std: {scores.std()*100:.2f}%")
