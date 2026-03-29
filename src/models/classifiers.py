from sklearn.base import BaseEstimator
from ..exception import ModelNotTrained



class BaseModel(BaseEstimator):
    def __init__ (self , name: str):
        self.name = name 
        self.is_trained = False 
       
    
    def __repr__(self):
       return (f"{self.__class__.__name__}(trained={self.is_trained})")
    




class HeartdiseaseClassifier(BaseModel):
    
    def __init__(self , n_estimators: int = 100):

        super().__init__("Heartdiseaseclassifier")
        self.n_estimators = n_estimators
        self._model = None
        self._features = 0 

    def fit(self , x ,y):
        from sklearn.ensemble import RandomForestClassifier
        self._model = RandomForestClassifier(n_estimators = self.n_estimators)
        self._model.fit(x , y)
        self.is_trained = True
        self._features = x.shape[1] 
        self.is_fitted_ = True

        return self
  

    def predict(self  , x):
        if not self.is_trained:
            raise ModelNotTrained(f"{self.name} is not trained. call fit() first")
        return self._model.predict(x)
    
    def predict_proba(self , x):
        if not self.is_trained:
            raise ModelNotTrained(f"{self.name} id not trained . call fit() first")
        return self._model.predict_proba(x) 
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(trained={self.is_trained} features = {self._features} )")
   