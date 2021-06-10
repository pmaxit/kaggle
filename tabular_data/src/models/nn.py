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
torch.manual_seed(0)

def residual_block(in_features, out_features, p_drop, non_linear = nn.ReLU(), *args, **kwargs):
    return nn.Sequential(
        nn.Dropout(p = p_drop),
        weight_norm(nn.Linear(in_features, out_features)),
        non_linear
    )

class TPSResidual(nn.Module):
    def __init__(self, num_class = 9, emb_szs = None, dropout = 0.3, linear_nodes=32, linear_out=16, emb_output=4, num_block=3):
        super().__init__()
        self.num_block = num_block
        
        self.final_module_list = nn.ModuleList()

        #self.embedding = nn.Embedding(feature_dictionary_size, emb_output)
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs]) #type: torch.nn.modules.container.ModuleList
        self.n_emb = sum(e.embedding_dim for e in self.embeds) # n_emb = 17 , type: int

        self.flatten = nn.Flatten()

        self.linear = weight_norm(nn.Linear(self.n_emb, linear_nodes))
        #self.linear = weight_norm(nn.Linear(emb_output * num_features, linear_nodes ))

        for res_num in range(self.num_block):
            self.non_linear = nn.ELU() if res_num %2 else nn.ReLU()
            self.lin_out = linear_out if res_num == (self.num_block - 1) else linear_nodes
            self.final_module_list.append(residual_block( self.n_emb + (res_num + 1) * linear_nodes, 
                                self.lin_out, dropout, self.non_linear))
        self.out = nn.Linear(linear_out, num_class)

        # non-linearity - activation function
        self.selu = nn.SELU()
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        # Embedding
        if self.n_emb != 0:
            x = [e(x[:,i]) for i,e in enumerate(self.embeds)] #take the embedding list and grab an embedding and pass in our single row of data.        
            x = torch.cat(x, 1) # concatenate it on dim 1 ## remeber that the len is the batch size
            x = self.dropout(x) # pass it through a dropout layer
        e = self.flatten(x)
        
        h1 = self.dropout(e)
        h1 = self.linear(h1)
        h1 = self.selu(h1)

        ri = torch.cat((e, h1), 1)
        for res_num in range(self.num_block):
            rx = self.final_module_list[res_num](ri)
            ri = torch.cat((ri, rx), 1)
        
        return F.softmax(self.out(rx), dim =-1)

class TPSResidualEstimator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lr_scheduler = LRScheduler(policy = ReduceLROnPlateau, monitor = 'valid_loss', mode = 'min', patience = 3, factor = 0.1, verbose = True)
        self.early_stopping = EarlyStopping(monitor='valid_loss', patience = 10, threshold = 0.0001, threshold_mode='rel', lower_is_better=True)
 
        self.lencoder = LabelEncoder()
    def fit(self, X, y=None):
        # here we will apply pipeline
        cat_cols = [c for c in X.columns if 'feature' in c]
        cat_szs = [max(X[col]) + 1 for col in cat_cols]
        emb_szs = [(size, min(50, (size+1)//3)) for size in cat_szs]

        self.net = NeuralNetClassifier(module=TPSResidual, module__emb_szs=emb_szs,
                          device = device, lr = 0.01, max_epochs = 50, 
                          callbacks = [self.lr_scheduler, self.early_stopping], 
                          batch_size=128,
                          optimizer__weight_decay=0.01,
                          iterator_train__shuffle=True,
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
        if isinstance(X, pd.DataFrame):
            X_test = X.values.astype('int64')
        else:
            raise NotImplementedError
            
        return self.net.predict_proba(X_test)
