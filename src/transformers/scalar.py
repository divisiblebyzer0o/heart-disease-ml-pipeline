from sklearn.base import BaseEstimator , TransformerMixin
import numpy as np


class RiskScorer(BaseEstimator , TransformerMixin):

    def fit(self , x , y = None):
        self.xmn = np.min(x , axis = 0)
        self.xmx = np.max(x , axis = 0)

        return self 
    
    def transform(self , x):

        
        result =     (x-self.xmn)/((self.xmx - self.xmn) + 1e-8) 
        
        return result