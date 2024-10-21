#   Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

#   Consts
D = 28 * 28
N = 2
C = 1
NCLASSES = 10
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#   Transpose func
class TransposeLayer(nn.Module):
    def __init__(self, linearLayer, bias = True) -> None:
        '''
        linearLayer -> the layer you want to transpose so if linearLayer is W this layer
        is W^T
        bias -> if True creates new bias term that is learned
        '''
        super().__init__()
        self.weight = linearLayer.weight
        if bias:
            self.bias = nn.Parameter(torch.tensor(
                linearLayer.weight.shape[1]
            ))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

#   View
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape) 

#   Encoder
linearLayer = nn.Linear(D, N, bias = False)
pcaEncoder = nn.Sequential(
    nn.Flatten(), linearLayer
)

#   Decoder
pcaDecoder = nn.Sequential(
    TransposeLayer(linearLayer, bias = False), View(-1, 1, 28, 28)
)

#   Sum
pcaModel = nn.Sequential(
    pcaEncoder, pcaDecoder
)

#   Orthogonality Constraint
nn.init.orthogonal_(linearLayer.weight)

mseLossFunc = nn.MSELoss()
def mseWithOrthoLoss(x, y):
    W = linearLayer.weight
    I = torch.eye(W.shape[0]).to(device)
    #   Plz learn to be a good autoencoder
    normalLoss = mseLossFunc(x, y) # l_mse(f(x), x)
    #   And try to keep your weights orthogonal
    reguralizationLoss = 0.1 * mseLossFunc(torch.mm(W, W.t()), I) # l_mse(WW^T, I)
    return normalLoss + reguralizationLoss

#   Dataset
class AutoEncoderDatset(Dataset):
    '''
    Dataset with (x, y) label -> Dataset with (x, x) labels
    '''
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset.__getitem__(idx)
        return (x, x)
