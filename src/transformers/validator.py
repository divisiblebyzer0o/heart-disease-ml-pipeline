from sklearn.base import BaseEstimator , TransformerMixin
from ..exception import FeatureMismatchError

class FeatureValidator(BaseEstimator , TransformerMixin):
    def fit(self , x , y = None):
        self.n_features = x.shape[1]
        return self
    
    def transform(self , x ):
        if self.n_features != x.shape[1]:
            raise FeatureMismatchError(f"{self.__class__.__name__} feature are not matched")
        return x 
          