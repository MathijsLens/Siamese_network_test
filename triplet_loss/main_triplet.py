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
from torch.utils.tensorboard import SummaryWriter
from triplet_loss import TripletLoss
from mnist_triplet import Triplet_MNIST
from triplet_siamese_model import Triplet_SiameseNetwork
from triplet_loss import clac_euclidian

def load_img_label(index, test_loader):
    for i, sample in enumerate(test_loader):
        if i== index:
            break
    batch_index=random.randint(0, len(sample[0]))
    print("picture ", batch_index, " from batch ", index)
    # plt.imshow(sample[0][batch_index].permute(1, 2, 0) )
    # plt.show()
    return sample[0][batch_index].unsqueeze(0), sample[1][batch_index]


def min_torch(config, test_embedding, querry_embedded, device):
    if config['test_batch']:
        dist=torch.Tensor(config['test_batch_size'], config['numb_classes']).to(device)
    else :
        dist=torch.Tensor(1, config['numb_classes']).to(device)
    for index, i in enumerate(querry_embedded):
        tensor_dist=(clac_euclidian(test_embedding, i))
        dist[:,index]=tensor_dist
    return torch.min(dist, 1)



"""returns a list containing true label in GT_L[0] and the detected label in GT_l[1]"""
def detect(config, model,device, anchor_img, anchor_label, querry_embedded):
    test_embedding=model(anchor_img.to(device))
    dist, label=min_torch(config, test_embedding, querry_embedded, device)
    true_list=[[], []]
    true_list[0]=anchor_label.to(device)
    true_list[1]=label
    return true_list
        
            
def test(config, device, train_dataset, test_loader):
    model = Triplet_SiameseNetwork()
    model.load_state_dict(torch.load("../saved_models/trained_model.pth")['model_state_dict'])
    model.eval()
    model.to(device)
    with torch.no_grad():    
        querry_embedded=[]            
        # get embedded querry's
        for img in train_dataset.querry:
            img=img.unsqueeze(0)
            querry_embedded.append(model(img.to(device)))
        acc=0
        
        
        #test on batches of images: 
        if config['test_batch']:       
            for batch_index, (anchor_img, anchor_label) in enumerate(test_loader):
                GT_l=detect(config, model,device, anchor_img.to(device), anchor_label, querry_embedded)
                for x,y in zip(GT_l[0], GT_l[1]):
                    if x==y: 
                        acc+=1
                # print("the acc after ", batch_index, "batch is ", acc/((batch_index+1)*config['test_batch_size']))
            acc/=len(test_loader)
            print("the total acc for test_set is ", acc/config['test_batch_size'])
        # test on single image
        else:
            index= random.randint(0, len(test_loader)) # sample a test image
            anchor_img, anchor_label= load_img_label(index, test_loader)
            GT_l=detect(config, model, device, anchor_img.to(device), anchor_label, querry_embedded)
            x,y = GT_l[0], GT_l[1]
            if x==y:
                print("correctly classifyed as ",torch.IntTensor.item(GT_l[0]))
            else: 
                print("wrongly classifyed as ",torch.IntTensor.item((GT_l[1])) , "instead of ",torch.IntTensor.item(GT_l[0]))
                
            
            




def train(config, model, device, train_loader, optim, epoch, writer=None):
    model.train() 
# triplet loss criterion: 
    criterion = TripletLoss()
    # trainloop: 
    
    for batch_index, (anchor_img, pos_img, neg_img, anchor_label) in enumerate(train_loader):
        anchor_img, pos_img, neg_img, anchor_label=anchor_img.to(device), pos_img.to(device),neg_img.to(device),anchor_label.to(device)
        optim.zero_grad()
        anchor_out  = model(anchor_img)
        pos_out  = model(pos_img)
        neg_out= model(neg_img)
        
        loss= criterion(anchor_out, pos_out, neg_out)
        loss.backward()
        optim.step()
        
        if batch_index % config['log_interval'] ==0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_index *len(anchor_img), len(train_loader.dataset), 
                                                                            100.0*batch_index /len(train_loader), loss.item()))
            if writer:
                writer.add_scalar('Training_loss', loss, global_step=epoch* len(train_loader)+batch_index)


def read_yaml():
    import yaml
    #read yaml file
    with open('../config.yaml') as file:
        config= yaml.safe_load(file)
    print(config)
    return config


def main():
    config=read_yaml()
    torch.manual_seed(config['seed'])

    if config['cuda']:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_dataset =Triplet_MNIST(config['path_dataset'], train=True, download=True)
    test_dataset = Triplet_MNIST(config['path_dataset'], train=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config['batch_size'], shuffle=True  )
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config['test_batch_size'], shuffle=True )

    writer=SummaryWriter(f'../runs/triplet/test')
    
    model = Triplet_SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config['learning_rate'])

    scheduler = StepLR(optimizer, step_size=1, gamma=config['gamma'])
    
    if config['train_net']:
        for epoch in range(1, config['epoch'] + 1):
            train(config, model, device, train_loader, optimizer, epoch, writer)
            test(config, device, train_dataset, test_loader )
            scheduler.step()
        if config['save_model']:
            torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict()
            }, "../saved_models/triplet_loss.pth")
        print("done training!")
    
    else: 
        test(config, device, train_dataset, test_loader )









if __name__ == '__main__':
    main()