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


def calc_dist(x1, x2):
    return np.power((x1-x2), 2).sum(1)

def min_dist(test_embedding, querry_embedded):
    dist=[]
    for i in querry_embedded:
        try: 
            dist.append(calc_dist(test_embedding, i))
        except:
            dist.append(clac_euclidian(test_embedding, i))
    min_value=min(dist)
    return dist.index(min_value), min_value


def min_torch(test_embedding, querry_embedded, device):
    dist=torch.Tensor(1000, 10).to(device)
    # dist=dist[:,None]
    for index, i in enumerate(querry_embedded):
        tensor_dist=(clac_euclidian(test_embedding, i))
        dist[:,index]=tensor_dist
    return torch.min(dist, 1)


def detect(model, test_img, test_label, querry_embedded):            
    test_embedding= model(test_img).cpu().numpy()
        
    label, distance=min_dist(test_embedding, querry_embedded)
    print("detected label = ", label, " The distance was ", distance, " the true label was ", test_label)
    
def detect_batch(model,device, anchor_img, anchor_label, querry_embedded):
    test_embedding=model(anchor_img.to(device))
    dist, label=min_torch(test_embedding, querry_embedded, device)
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
        for batch_index, (anchor_img, anchor_label) in enumerate(test_loader):
            GT_l=detect_batch(model,device, anchor_img.to(device), anchor_label, querry_embedded)
            for x,y in zip(GT_l[0], GT_l[1]):
                if x==y: 
                    acc+=1
            print("the acc after ", batch_index, "batch is ", acc/((batch_index+1)*1000))
        acc/=len(test_loader)
        print("the total acc for test_set is ", acc/1000.0)

def train(model, device, train_loader, optim, epoch, writer=None):
    model.train() 
# triplet loss criterion: 
    criterion = TripletLoss()
    # trainloop: 
    
    for batch_index, (config, anchor_img, pos_img, neg_img, anchor_label) in enumerate(train_loader):
        anchor_img, pos_img, neg_img, anchor_label=anchor_img.to(device), pos_img.to(device),neg_img.to(device),anchor_label.to(device)
        optim.zero_grad()
        anchor_out  = model(anchor_img)
        pos_out  = model(pos_img)
        neg_out= model(neg_img)
        
        loss= criterion(anchor_out, pos_out, neg_out)
        loss.backward()
        optim.step()
        
        if batch_index % config.log_interval ==0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_index *len(anchor_img), len(train_loader.dataset), 
                                                                            100.0*batch_index /len(train_loader), loss.item()))
            if writer:
                writer.add_scalar('Training_loss', loss, global_step=epoch* len(train_loader)+batch_index)


def read_yaml():
    import yaml
    #read yaml file
    with open('config.yaml') as file:
        config= yaml.safe_load(file)
    print(config)
    return config


def main(Train=True):
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

    writer=SummaryWriter(f'runs/siamese/test')
    
    model = Triplet_SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config['learning_rate'])

    scheduler = StepLR(optimizer, step_size=1, gamma=config['gamma'])
    
    if config['train_net']:
        for epoch in range(1, config['epoch'] + 1):
            train(model, device, train_loader, optimizer, epoch, writer)
            scheduler.step()
        if config['save_model']:
            torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict()
            }, "../saved_models/triplet_loss.pth")
    
    else: 
        test(config, device, train_dataset, test_loader )









if __name__ == '__main__':
    main(True)