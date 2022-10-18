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

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # get resnet model:
        self.resnet=torchvision.models.resnet18(pretrained=False)

        #overwrite resnet first layer to accept grayscale (1, x, x) ipv (3, x, x )

        self.resnet.conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3,3), bias=False)
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer) take all but last layer
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
    
        
        self.fc_in_features = 512  # what the resnet outputs is

        # add fc layer:
        self.fc=nn.Sequential(nn.Linear(self.fc_in_features *2, 256), # for 2 pictures thats why times 2 
        nn.ReLU(inplace=True),
        nn.Linear(256, 1))

        self.sigmoid = nn.Sigmoid()  # compress output to 0 - 1

        # init weights 
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output=self.resnet(x)
        output=output.view(output.size()[0], -1)
        return output
    
    def forward(self, in1, in2):
        out1=self.forward_once(in1)  # give first input
        out2=self.forward_once(in2)  # give second input

        output=torch.cat((out1, out2), 1) # concat the embeddings

        output= self.fc(output)  # pass throu fc

        output = self.sigmoid(output) # pass throu sigmoid 
        return output