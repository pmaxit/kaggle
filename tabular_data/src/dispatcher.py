from sklearn import ensemble
from src.models.random_forest import RandomForest
from src.models.nn import TPSResidualEstimator
from src.models.nn_advanced import TPSResidualEstimatorAdv
from src.models.tabular import TabularEstimator

MODELS = {
    'randomforest': RandomForest(),
    'extratrees': ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    'nn': TPSResidualEstimator(),
    'nnadv': TPSResidualEstimatorAdv(),
    'tab': TabularEstimator()
}