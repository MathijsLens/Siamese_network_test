from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
import random, copy, argparse
import torch
import numpy as np
from PIL import Image
import PIL
import matplotlib.pyplot as plt 

num_classes=10
# data class for MNIST dataset
class Triplet_MNIST(Dataset):
    def __init__(self, root, train, download=False):
        super(Triplet_MNIST, self).__init__()
        self.dataset= datasets.MNIST(root, train, download=download)
        self.data= self.dataset.data.unsqueeze(1).clone()   # ads a channel dimention to the images 
        self.group_examples()
        self.train=train
        self.querry=self.get_Querry()      
        
    
    
    # used to group examples based on class
       
    def group_examples(self):
        np_arr= np.array(self.dataset.targets.clone())
        
        self.grouped_examples={}
        for i in range(0, num_classes):
            self.grouped_examples[i]=np.where((np_arr==i))[0]
            
            
    def __len__(self):
        return self.data.shape[0]
    
    
    "returns list containing img data for each class"
    def get_Querry(self):
        index_querry=[]
        for i in range(0, num_classes):
            index_querry.append(self.grouped_examples[i][0])
        querry=[]
        for i in index_querry:
            querry.append(self.data[i])
        return querry
    
    def __getitem__(self, index):
        anchor_label=self.dataset.targets[index].item()
        anchor_img= self.data[index]
        if self.train:
            
            
            pos_index_list=random.randint(0, len(self.grouped_examples[anchor_label])-1) # random index in pos list
            pos_index= self.grouped_examples[anchor_label][pos_index_list].item()
            while (pos_index== index):
                pos_index_list=random.randint(0, len(self.grouped_examples[anchor_label])-1) # random index in pos list
                pos_index= self.grouped_examples[anchor_label][pos_index_list].item()
                
            pos_img=self.data[pos_index]
            
            neg_index=random.randint(0, self.__len__()-1)  # random index on zhole dataset
            while(self.dataset.targets[neg_index].item()==anchor_label):
                neg_index=random.randint(0, self.__len__()-1)
            
            neg_img = self.data[neg_index]
                
            
            return anchor_img, pos_img, neg_img, anchor_label
        else :
            return anchor_img, anchor_label
    
def debug():
    mnist_data=Triplet_MNIST('../datasets', train=True, download=True)
    a, p, n, l=mnist_data.__getitem__(80)
    image_list=mnist_data.get_Querry()
    plt.imshow(image_list[9].permute(1,2,0))
    plt.show()
    
    # print(l)
    
    # plt.imshow(a.permute(1, 2, 0) )
    # plt.show()
    # plt.imshow(p.permute(1, 2, 0))
    # plt.show()
    # plt.imshow(n.permute(1, 2, 0))
    # plt.show()
    
    
    
    
    
if __name__=='__main__':
    debug()
    