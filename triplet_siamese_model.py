import matplotlib.pyplot as plt
import numpy as np
import random, copy, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR

class Triplet_SiameseNetwork(nn.Module):
    def __init__(self, embed_dim=128):
        super(Triplet_SiameseNetwork, self).__init__()
        # get resnet model:
        self.resnet=torchvision.models.resnet18(pretrained=False)

        #overwrite resnet first layer to accept grayscale (1, x, x) ipv (3, x, x )

        self.resnet.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias=False)
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer) take all but last layer
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
    
        
        self.fc_in_features = 512  # what the resnet outputs is

        # add fc layer:
        self.fc=nn.Sequential(nn.Linear(self.fc_in_features, 256), # for 2 pictures thats why times 2 
        nn.ReLU(inplace=True),
        nn.Linear(256,embed_dim ))

        # init weights 
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        output=self.resnet(x.float())
        output=output.view(-1,self.fc_in_features)
        output=self.fc(output)
        return output
    