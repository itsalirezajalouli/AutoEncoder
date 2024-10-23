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

#   Noise
normal = torch.distributions.Normal(0, 0.5)

def addNoise(x, device = 'cpu'):
    return x + normal.sample(sample_shape = torch.Size(x.shape)).to(device)

class AdditiveGaussianNoise(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        if self.training:
            return addNoise(x, device = device)

linearLayer = nn.Linear(D, N, bias = False)
def getLayer(inSize, outSize):
    return nn.Sequential(
        nn.Linear(inSize, outSize),
        nn.BatchNorm1d(outSize),
        nn.ReLU(),
    )

#   Encoder
autoEncoder = nn.Sequential(
    nn.Flatten(),
    # AdditiveGaussianNoise(),
    nn.Dropout(0.2),
    getLayer(D, D//2),
    nn.Dropout(0.2),
    getLayer(D//2, D//3),
    nn.Dropout(0.2),
    getLayer(D//3, D//4),
    nn.Dropout(0.2),
    nn.Linear(D//4, N),
)

#   Decoder
autoDecoder = nn.Sequential(
    # TransposeLayer(linearLayer, bias = False), View(-1, 1, 28, 28)
    getLayer(N, D//4),
    nn.Dropout(0.2),
    getLayer(D//4, D//3),
    nn.Dropout(0.2),
    getLayer(D//3, D//2),
    nn.Dropout(0.2),
    nn.Linear(D//2, D),
    View(-1, 1, 28, 28)
)

#   Sum
model = nn.Sequential(
    autoEncoder, autoDecoder
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

results = train_network(model, mseLossFunc, trainLoader,
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

#   Plot
projected, labels = encodeBatch(autoEncoder, testSetXY)
sns.scatterplot(x = projected[:, 0], y = projected[:, 1],
                hue = [str(l) for l in labels],
                hue_order = [str(i) for i in range(10)], legend = 'full')
plt.savefig('./plts/scatter.png')

def showEncodeDecode(encodeDecode, x, pltname):
    encodeDecode = encodeDecode.eval()
    encodeDecode = encodeDecode.cpu()
    with torch.no_grad():
        xRecon = encodeDecode(x.cpu())
    _, axarr = plt.subplots(1, 2)
    axarr[0].imshow(x.numpy()[0, :])
    axarr[1].imshow(xRecon.numpy()[0, 0, :])
    plt.savefig(f'{pltname}.png')

showEncodeDecode(model, testSetXY[0][0], './plts/image1')
showEncodeDecode(model, addNoise(testSetXY[2][0]), './plts/image2')
showEncodeDecode(model, addNoise(testSetXY[10][0]), './plts/image3')
showEncodeDecode(model, testSetXY[25][0], './plts/image4')
