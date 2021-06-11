# idea of this class is to have a function which calls fit, transform
import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import torch.nn.functional as F

from skorch import NeuralNetClassifier, NeuralNet
from skorch.callbacks import EpochScoring
from sklearn.preprocessing import LabelEncoder
from skorch.callbacks import LRScheduler, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

import warnings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

torch.manual_seed(0)
def residual_block(in_features, out_features, batch_norm, p_drop, non_linear = nn.ReLU(), *args, **kwargs):       
        net = nn.Sequential(
            nn.Dropout(p = p_drop),
            weight_norm(nn.Linear(in_features, out_features)),
            non_linear) 
        
        if batch_norm:
            net = nn.Sequential(nn.BatchNorm1d(in_features),
                                nn.Dropout(p = p_drop),
                                nn.Linear(in_features, out_features),
                                non_linear)
        return net



class TPSResidual(nn.Module):
    def __init__(self, 
                 num_features = 75, 
                 num_class = 9, 
                 feature_dictionary_size = 360, 
                 batch_norm = False,
                 dropout = 0.3, 
                 linear_nodes = 32, 
                 linear_out = 16, 
                 emb_output = 4, 
                 num_block = 3, **kwargs):
        super(TPSResidual, self).__init__()
        self.num_block = num_block
        self.final_module_list = nn.ModuleList()
    
        
        self.embedding = nn.Embedding(feature_dictionary_size, emb_output)
        self.flatten = nn.Flatten()

        self.linear = weight_norm(nn.Linear(emb_output * num_features, linear_nodes))
        torch.nn.init.xavier_uniform(self.linear.weight)
        
        for res_num in range(self.num_block):
            self.non_linear = nn.ELU() if res_num % 2 else nn.ReLU()
            self.lin_out = linear_out if res_num == (num_block-1) else linear_nodes
            self.final_module_list.append(residual_block(emb_output * num_features + (res_num + 1) * linear_nodes, self.lin_out, batch_norm, dropout, self.non_linear))
        
        #self.bn = nn.BatchNorm1d(linear_out)
        self.out = nn.Linear(linear_out, num_class)
        
        # nonlinearity - activation function
        self.selu = nn.SELU()
        
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        x = torch.tensor(x).to(torch.int64)
        
        # Embedding 
        e = self.embedding(x)
        e = self.flatten(e)
        
        h1 = self.dropout(e)
        h1 = self.linear(h1)
        h1 = self.selu(h1)
        
        ri = torch.cat((e, h1), 1)
        
        for res_num in range(self.num_block):          
            rx = self.final_module_list[res_num](ri)
            ri = torch.cat((ri, rx), 1)
        # rx = self.bn(rx)
        return  self.out(rx)


class TPSResidualEstimatorAdv(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lr_scheduler = LRScheduler(policy = ReduceLROnPlateau, monitor = 'valid_loss', mode = 'min', patience = 3, factor = 0.1, verbose = True)
        self.early_stopping = EarlyStopping(monitor='valid_loss', patience = 10, threshold = 0.0001, threshold_mode='rel', lower_is_better=True)
 
        self.lencoder = LabelEncoder()

    def fit(self, X, y=None):
        # here we will apply pipeline
        cat_cols = [c for c in X.columns if 'feature' in c]
        cat_szs = [max(X[col]) + 1 for col in cat_cols]
        emb_szs = [(size, min(50, (size+1)//3)) for size in cat_szs]

        self.net = NeuralNetClassifier(module=TPSResidual,
                          device = device, lr = 0.01, max_epochs = 50, 
                          callbacks = [self.lr_scheduler, self.early_stopping], 
                          batch_size=64
                         )
        
        # if X is a dataframe, we need to convert to numpy array
        if isinstance(X, pd.DataFrame):
            X_train = X.values.astype('int64')
            y_train = self.lencoder.fit_transform(y).astype('int64')
        else:
            raise NotImplementedError

        if X_train is not None:
            self.net.fit(X_train, y_train)

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X_test = X.values.astype('int64')
        else:
            raise NotImplementedError

        y_pred = self.net.predict(X_test)
        return self.lencoder.inverse_transform(y_pred)
    
    def predict_proba(self, X):
        return self.net.predict_proba(X)
