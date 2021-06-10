from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

class TabularEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='target'):
        self.trainer_config = TrainerConfig(
            auto_lr_find=True,
            batch_size=1024,
            max_epochs=100,
            #gpus=0 # index of the GPU to use.
        )
        self.target_col = target_col
        self.optimizer_config = OptimizerConfig()
        self.model_config = CategoryEmbeddingModelConfig(
            task='classification',
            layers='256-64',
            activation='LeakyReLU',
            learning_rate=1e-3
        )

    def combine_data(self, X, y):
        if y is None:
            raise NotImplementedError

        data = X.copy()
        data[self.target_col] = y
        return data

    def init_model(self, data):
        cat_cols = [c for c in data.columns if c != self.target_col]
        data_config = DataConfig(
            target = [self.target_col],
            categorical_cols= cat_cols,
            num_workers=16,
        )

        tabular_model = TabularModel(
            data_config = data_config,
            model_config = self.model_config,
            optimizer_config = self.optimizer_config,
            trainer_config = self.trainer_config
        )
        return tabular_model

    def prepare_data(self,X,y,**kwargs):
        X_train, X_test, y_train, y_test = train_test_split(X,y, **kwargs)
        return X_train,X_test, y_train, y_test


    def fit(self, X,y=None,full=False):
        ''' full: to use full training data to fit the model '''

        # we will spit the data for validation loss
        if not full:
            X_train, X_val, y_train, y_val = self.prepare_data(X, y, test_size=0.2, shuffle=True, random_state=42)
            data_train = self.combine_data(X_train, y_train)
            data_val = self.combine_data(X_val, y_val)
            
            self.clf = self.init_model(data_train)

            self.clf.fit(train=data_train, validation=data_val)
        else:
            # use the entire data for training
            data_train = self.combine_data(X, y)
            self.clf = self.init_model(data_train)
            self.clf.fit(train=data_train)

    def predict(self, X):
        if self.clf is None:
            raise Exception('Fit not called !')
        return self.clf.predict(X)['prediction']
    
    def predict_proba(self, X):
        if self.clf is None:
            raise Exception('Fit not called !')
        
        temp = self.clf.predict(X, ret_logits=True)
        return temp[[c for c in temp.columns if 'probability' in c]].values