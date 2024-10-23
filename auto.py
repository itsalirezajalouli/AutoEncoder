#   Imports
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from idlmam import train_network
import torchvision.transforms as T 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset

#   Consts
D = 28 * 28
N = 2
C = 1
NCLASSES = 10
batchSize = 128
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
        return F.linear(x, self.weight.t(), self.bias)
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

#   Dataset & Loader
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

trainSet = AutoEncoderDatset(MNIST('../mnist/', True, T.ToTensor(), download = True))
testSetXY = MNIST('../mnist/', False, T.ToTensor(), download = True)
testSetXX = AutoEncoderDatset(testSetXY)
trainLoader = DataLoader(trainSet, batchSize, True)
testLoader = DataLoader(testSetXX, batchSize)

results = train_network(pcaModel, mseLossFunc, trainLoader,
                        testLoader, epochs = 10, device = device)
print(results)

#   Encode batch
def encodeBatch(encoder, dataset2encode):
    projected = []
    labels = []

    encoder = encoder.eval()
    encoder = encoder.cpu()

    with torch.no_grad():
        for x, y in DataLoader(dataset2encode, batchSize):
            z = encoder(x.cpu())
            projected.append(z.numpy())
            labels.append(y.cpu().numpy().ravel())
    projected = np.vstack(projected)
    labels = np.hstack(labels)
    return projected, labels

projected, labels = encodeBatch(pcaEncoder, testSetXY)
sns.scatterplot(x = projected[:, 0], y = projected[:, 1],
                hue = [str(l) for l in labels],
                hue_order = [str(i) for i in range(10)], lengend = 'full')

def showEncodeDecode(encodeDecode, x):
    encodeDecode = encodeDecode.eval()
    encodeDecode = encodeDecode.cpu()
    with torch.no_grad():
        xRecon = encodeDecode(x.cpu())
    _, axarr = plt.subplots(1, 2)
    axarr[0].imshow(x.numpy()[0, :])
    axarr[1].imshow(xRecon.numpy()[0, 0, :])
