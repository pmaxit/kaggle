import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.optim as optim
import torch.nn.functional as F

from skorch import NeuralNetClassifier, NeuralNet
from skorch.callbacks import EpochScoring
from skorch.callbacks import LRScheduler, EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau


import warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

feature_dictionary_size= 360
num_features = 75

def residual_block(in_features, out_features, p_drop, non_linear = nn.ReLU(), *args, **kwargs):
    return nn.Sequential(
        nn.Dropout(p = p_drop),
        weight_norm(nn.Linear(in_features, out_features)),
        non_linear
    )

class TPSResidual(nn.Module):
    def __init__(self, num_class = 9, dropout = 0.3, linear_nodes=32, linear_out=16, emb_output=4, num_block=3):
        super().__init__()
        self.num_block = num_block
        self.final_module_list = nn.ModuleList()

        self.embedding = nn.Embedding(feature_dictionary_size, emb_output)
        self.flatten = nn.Flatten()

        self.linear = weight_norm(nn.Linear(emb_output * num_features, linear_nodes ))

        for res_num in range(self.num_block):
            self.non_linear = nn.ELU() if res_num %2 else nn.ReLU()
            self.lin_out = linear_out if res_num == (self.num_block - 1) else linear_nodes
            self.final_module_list.append(residual_block(emb_output * num_features + (res_num + 1) * linear_nodes, 
                                self.lin_out, dropout, self.non_linear))
        self.out = nn.Linear(linear_out, num_class)

        # non-linearity - activation function
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
        
        return F.softmax(self.out(rx), dim =-1)

lr_scheduler = LRScheduler(policy = ReduceLROnPlateau, monitor = 'valid_loss', mode = 'min', patience = 3, factor = 0.1, verbose = True)
early_stopping = EarlyStopping(monitor='valid_loss', patience = 10, threshold = 0.0001, threshold_mode='rel', lower_is_better=True)

net = NeuralNetClassifier(TPSResidual, device = device, lr = 0.001, max_epochs = 50, callbacks = [lr_scheduler, early_stopping])