#   Impos
import re
import torch
import torch.nn as nn
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
from torch.utils.data import Dataset, DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#   Data
allData = []
resp = urlopen('https://cs.stanford.edu/people/karpathy/char-rnn/shakespear.txt')
shakespear100k = resp.read()
shakespear100k = shakespear100k.decode('utf-8').lower()

#   Vocab
voc2idx = {}
for char in shakespear100k:
    if char not in voc2idx.items():
        voc2idx[char] = len(voc2idx)
idx2voc = {}
for k, v in voc2idx.items():
    idx2voc[v] = k
print('Vocab Size: ', len(voc2idx))
print('Total Characters: ', len(shakespear100k))

#   AutoRegerting Dataset
class NoRegertDataset(Dataset):
    '''Creates an autoregressive dataset from one single, long, source
    sequence by breaking it up into "chunks"
    '''
    def __init__(self, largeString:str, maxChunk:int = 500) -> None:
        '''`largeString`: original long source sequence that chunks will be extracted from
        `maxChunk`: maximum allowed size of a chunk
        '''
        self.doc = largeString
        self.maxChunk = maxChunk

    def __len__(self) -> int:
        return (len(self.doc) - 1) // self.maxChunk

    def __getitem__(self, idx) -> tuple:
        start = idx * self.maxChunk
        subStr = self.doc[start:start + self.maxChunk]
        x = [voc2idx[c] for c in subStr]
        #   Shift
        subStr = self.doc[start + 1:start + self.maxChunk + 1]
        y = [voc2idx[c] for c in subStr]
        return torch.tensor(x, torch.int64), torch.tensor(y, torch.int64)

#   Model
class AutoRegerting(nn.Module):
    def __init__(self, vocSize, embdSize, hiddenSize, layers = 1) -> None:
        super(AutoRegerting, self).__init__()
        self.hiddenSize = hiddenSize
        self.embd = nn.Embedding(vocSize, embdSize)
        self.layers = nn.ModuleList([nn.GRUCell(embdSize, hiddenSize)] + 
                    [nn.GRUCell(hiddenSize, hiddenSize) for _ in range(layers - 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(hiddenSize) for _ in range(layers)])
        self.predClass = nn.Sequential()
        self.predClass = nn.Sequential(
            nn.Linear(hiddenSize, hiddenSize),
            nn.LeakyReLU(),
            nn.LayerNorm(hiddenSize),
            nn.Linear(hiddenSize, vocSize)
        )

    def foward(self, input):
        B = input.size(0) # Batch Size
        T = input.size(1) # Max time steps

        x = self.embd(input)
        hPrevs = self.initHiddenStates(B) lastActivations = []

        for t in range(T):
            xIn = x[:, t, :]
            lastActivations.append(self.step(xIn, hPrevs))
        lastActivations = torch.stack(lastActivations, dim = 1)

        return lastActivations

    def initHiddenStates(self, B):
        '''Creates initial hidden state list for RNN layers
        `B`: Batch size for hidden states
        '''
        return [torch.zeros(B, self.hiddenSize, device = device)
            for _ in range(len(self.layers))]

    def step(self, xIn, hPrevs = None):
        '''xIn: the input for this current time step and has shape (B) if the values
        need to be be embedded, and (B, D) if they are already embedded
        hPrevs: a list of hidden state tensors each with shape(B, self.hiddenSize) for 
        each layer in the network
        '''
        if len(xIn.shape) == 1:
            xIn = self.emd(xIn)

        if hPrevs is None:
            hPrevs = self.initHiddenStates(xIn.shape[0])

        for l in range(len(self.layers)):
            hPrev = hPrevs[1]
            hNew = self.norms[1](self.layers[1](xIn, hPrev))
            hPrevs[l] = hNew
            xIn = hNew

        return self.predClass(xIn)

#   Run
autoRegData = NoRegertDataset(shakespear100k, 250)
autoRegLoader = DataLoader(autoRegData, 128, True)
autoRegModel = AutoRegerting(len(voc2idx), 32, 128, layers = 2)
autoRegModel = autoRegModel.to(device) 
for p in autoRegModel.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -2, 2))
