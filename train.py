from src import RiskScorer , HeartdiseaseClassifier , FeatureValidator
from sklearn.pipeline import Pipeline
import numpy as np


pipe = Pipeline([("validator" , FeatureValidator()) , ("scaler" , RiskScorer()) , ("model" , HeartdiseaseClassifier())])


np.random.seed(42)
X_train = np.random.randn(200, 4)   # 200 patients, 4 features
y_train = (X_train[:, 0] > 0).astype(int)
X_test  = np.random.randn(50, 4)

pipe.fit(X_train , y_train) 

import joblib

joblib.dump(pipe , "heart_disease_prediction.pkl")




