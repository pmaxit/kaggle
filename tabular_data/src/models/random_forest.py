from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import ensemble
from sklearn.pipeline import make_pipeline

from src.features import preprocessor

class RandomForest(BaseEstimator, TransformerMixin):
    def __init__(self,*args, **kwargs):
        self.clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2,**kwargs)
        self.final_pipeline = make_pipeline(preprocessor, self.clf)

    def fit(self, X , y = None):
        self.final_pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.final_pipeline.predict(X)