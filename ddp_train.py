import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

import argparse
from datetime import datetime
import os
import torch.multiprocessing as mp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N') #system in the istributed systems. a single systems has multiple gpus can be called as a node
    parser.add_argument('-g', '--gpus', default=3, type=int,
                        help='number of gpus per node') #number of gpus
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes') # unique identification number for processes in each node
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-bz', '--batch_size', default=100, type=int, metavar='N',
                        help='number of batches')
    
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    print(f'arguments: {args}')
    # following parameters are set according to  https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
    os.environ['MASTER_ADDR'] = 'localhost'#'130.85.90.70'
    os.environ['MASTER_PORT'] = '12355'
    mp.spawn(train, nprocs=args.gpus, args=(args,))

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def train(gpu, args):

    rank = args.nr * args.gpus + gpu #0*3+3 =3
    dist.init_process_group(
    	backend='nccl',
   		init_method='env://',
    	world_size=args.world_size,
    	rank=rank
    )

    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)

    start = datetime.now()
    print(f'start time: {start}')
    total_step = len(train_loader)
    for epoch in range(args.epochs):
#         print(f'strating epoch {epoch}')
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, 
                    args.epochs, 
                    i + 1, 
                    total_step,
                    loss.item()))
    end = datetime.now()
    print(f'end time: {end}')
    print("Training complete in: " + str(end - start))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()