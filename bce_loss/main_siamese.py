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
            if writer:
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
    
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=config['batch_size'], shuffle=True  )
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=config['test_batch_size'], shuffle=True )

    writer=SummaryWriter(f'../runs/bce/test')
    
    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config['learning_rate'])

    scheduler = StepLR(optimizer, step_size=1, gamma=config['gamma'])
    
    #training
    if config['train_net']:
        for epoch in range(1, config['epoch'] + 1):
            train(config, model, device, train_loader, optimizer, epoch, writer)
            test(model, device, test_loader)
            scheduler.step()

        if config['save_model']:
            torch.save({"model_state_dict": model.state_dict(),
                    "optimzier_state_dict": optimizer.state_dict()
                }, "../saved_models/BCE_loss.pth")
    else: 
        test(model, device, test_loader)







def debug():
    #training settings:
    parser= argparse.ArgumentParser(description='pytorch siamese network example')
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                        'pin_memory': True,
                        'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = mnist_dataload.APP_MATCHER('../datasets', train=True, download=True)
    test_dataset = mnist_dataload.APP_MATCHER('../datasets', train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
    model = siamese_model.SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    # Display image and label.
 
    # train_features, train_labels = next(iter(train_loader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")

if __name__ == '__main__':
    main()
    # debug()