from turtle import distance, forward
import torch.nn as nn
import torch

def clac_euclidian(x1, x2):
        return (x1-x2).pow(2).sum(1)
    
    
class TripletLoss(nn.Module):
    def __init__(self, margin=10.0):
        super(TripletLoss, self).__init__()
        self.margin=margin
        
    
    
    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor):
        distance_pos=clac_euclidian(anchor, pos)
        distance_neg= clac_euclidian(anchor, neg)
        losses= torch.relu(distance_pos-distance_neg + self.margin)
        return losses.mean()
    
