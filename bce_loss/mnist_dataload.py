from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T
import random, copy, argparse
import torch
import numpy as np


num_classes=10
# data class for MNIST dataset
class APP_MATCHER(Dataset):
    def __init__(self, root, train, download=False):
        super(APP_MATCHER, self).__init__()
        self.dataset= datasets.MNIST(root, train, download=download)
        self.data= self.dataset.data.unsqueeze(1).clone()   # ads a channel dimention to the images 
        
        self.group_examples()
    
    
    # used to group examples based on class
       
    def group_examples(self):
        np_arr= np.array(self.dataset.targets.clone())
        
        self.grouped_examples={}
        for i in range(0, num_classes):
            self.grouped_examples[i]=np.where((np_arr==i))[0]
            
            
    def __len__(self):
        return self.data.shape[0]
    
    
    """
            For every example, we will select two images. There are two cases, 
            positive and negative examples. For positive examples, we will have two 
            images from the same class. For negative examples, we will have two images 
            from different classes.
            Given an index, if the index is even, we will pick the second image from the same class, 
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn 
            the similarity between two different images representing the same class. If the index is odd, we will 
            pick the second image from a different class than the first image.
    """
    def __getitem__(self, index):
        # pick some random class:
        selected_class= random.randint(0, num_classes-1)
        
        # select random index in that class
        
        random_index_1= random.randint(0, self.grouped_examples[selected_class].shape[0]-1) # number between 0 and numb of examples for that class
        index_1= self.grouped_examples[selected_class][random_index_1]
        image_1= self.data[index_1].clone().float()   # get first img
        
        if index %2==0:
            # even: 
            random_index_2=random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            while random_index_2==random_index_1:
                random_index_2=random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            index_2= self.grouped_examples[selected_class][random_index_2]
            target= torch.tensor(1, dtype=torch.float) # create annotation 1 means  same class
            image_2= self.data[index_2].clone().float()
            
            
        else:
            class_2=random.randint(0, num_classes-1)
            while class_2==selected_class:
                class_2 = random.randint(0, num_classes-1)  # not same class as first image
                
            random_index_2= random.randint(0, self.grouped_examples[class_2].shape[0]-1) # get random index for other class   
            index_2= self.grouped_examples[class_2][random_index_2]
            target= torch.tensor(0, dtype=torch.float) # create annotation 0 means different class
            image_2= self.data[index_2].clone().float()
        return image_1, image_2, target