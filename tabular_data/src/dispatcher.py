from sklearn import ensemble
from src.models.random_forest import RandomForest
from src.models.nn import TPSResidualEstimator

MODELS = {
    'randomforest': RandomForest(),
    'extratrees': ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    'nn': TPSResidualEstimator()
}