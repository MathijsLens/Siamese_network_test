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
from mnist_dataload import  MNIST_BCE
from siamese_model import SiameseNetwork


def train(config, model, device, train_loader, optim, epoch, writer=None):
    model.train()
    # criterion: LoSS criterion here BCE LOSS
    criterion= nn.BCELoss()
    # trainloop: 
    for batch_index, (image_1, image_2, targets) in enumerate(train_loader):        
        image_1, image_2, targets=image_1.to(device), image_2.to(device), targets.to(device)
        optim.zero_grad()
        outputs  = model(image_1, image_2).squeeze()
        loss= criterion(outputs, targets)
        loss.backward()
        optim.step()
        
        predictions= outputs
        num_correct= (torch.round(predictions)==targets).sum()
        running_acc=float(num_correct)/float(image_1.shape[0])
        
        if batch_index % config['log_interval'] ==0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_index *len(image_1), len(train_loader.dataset), 
                                                                            100.0*batch_index /len(train_loader), loss.item()))
            
            writer.add_scalar('Training_loss', loss, global_step=batch_index)
            writer.add_scalar('Training accuracy', running_acc, global_step=batch_index)


       
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()
    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

        # for the 1st epoch, the average loss is 0.0001 and the accuracy 97-98%
        # using default settings. After completing the 10th epoch, the average
        # loss is 0.0000 and the accuracy 99.5-100% using default settings.
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        

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

    
    train_dataset = MNIST_BCE(config['path_dataset'], train=True, download=True)
    test_dataset =  MNIST_BCE(config['path_dataset'], train=False)
    # train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config['batch_size'], shuffle=True  )
    # test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config['test_batch_size'], shuffle=True )
    writer=SummaryWriter(f'../runs/bce/test')
    
    model = SiameseNetwork().to(device)
    
    
    # search for hyperparams: 
    if config['hypersearch']:
        for batch_s in config['batch_size']: 
            for lr in config['learning_rate']:
                writer=SummaryWriter(f'../runs/bce/batchsize{batch_s} lr {lr}')
                train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_s, shuffle=True  )
                test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config['test_batch_size'], shuffle=True )
                model = SiameseNetwork().to(device)
                optimizer = optim.Adadelta(model.parameters(), lr=lr)
                scheduler = StepLR(optimizer, step_size=1, gamma=config['gamma'])
                for epoch in range(0, config['epoch']):
                    train(config, model, device, train_loader, optimizer, epoch, writer)
                    test(model, device, test_loader)
                    scheduler.step()
                    
                
    #train 
    elif config['train_net']:
        
        optimizer = optim.Adadelta(model.parameters(), lr=config['learning_rate'])
        scheduler = StepLR(optimizer, step_size=1, gamma=config['gamma'])
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config['batch_size'], shuffle=True  )
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config['test_batch_size'], shuffle=True )
        for epoch in range(1, config['epoch'] + 1):
            train(config, model, device, train_loader, optimizer, epoch, writer)
            test(model, device, test_loader)
            scheduler.step()

        if config['save_model']:
            torch.save({"model_state_dict": model.state_dict(),
                    "optimzier_state_dict": optimizer.state_dict()
                }, "../saved_models/BCE_loss.pth")
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config['batch_size'], shuffle=True  )
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config['test_batch_size'], shuffle=True ) 
        test(model, device, test_loader)




if __name__ == '__main__':
    main()
    # debug()