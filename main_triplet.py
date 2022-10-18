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
        
            
def test(device, train_dataset, test_loader):
    model = Triplet_SiameseNetwork()
    model.load_state_dict(torch.load("trained_model.pth")['model_state_dict'])
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

def train(args, model, device, train_loader, optim, epoch, writer=None):
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
        
        if batch_index % args.log_interval ==0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( epoch, batch_index *len(anchor_img), len(train_loader.dataset), 
                                                                            100.0*batch_index /len(train_loader), loss.item()))
            if writer:
                writer.add_scalar('Training_loss', loss, global_step=epoch* len(train_loader)+batch_index)
            if args.dry_run:
                break



def main(Train=True):
    #training settings:
    parser= argparse.ArgumentParser(description='pytorch siamese network example')
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
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
    parser.add_argument('--save-model', action='store_true', default=True,
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
    
    train_dataset =Triplet_MNIST('../datasets', train=True, download=True)
    test_dataset = Triplet_MNIST('../datasets', train=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    writer=SummaryWriter(f'runs/siamese/test')
    
    model = Triplet_SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    if Train:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, writer)
            scheduler.step()
        
        torch.save({"model_state_dict": model.state_dict(),
                "optimzier_state_dict": optimizer.state_dict()
            }, "trained_model.pth")
    
    else: 
        # model = Triplet_SiameseNetwork()
        # model.load_state_dict(torch.load("trained_model.pth")['model_state_dict'])
        # model.eval()
        # with torch.no_grad():    
        #     querry_embedded=[]            
        #     # get embedded querry's
        #     for img in train_dataset.querry:
        #         img=img.unsqueeze(0)
        #         querry_embedded.append(model(img).cpu().numpy())
        #     detect(model,train_dataset.data[9].unsqueeze(0), train_dataset.dataset.targets[9], querry_embedded)
        test(device, train_dataset, test_loader )









if __name__ == '__main__':
    main(False)