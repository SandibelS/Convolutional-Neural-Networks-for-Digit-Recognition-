# Una red convolucional con la siguiente configuración: INPUT -> CONV -> RELU -> FC -> RELU -> FC

import torch

import torch.nn as nn
import torch.nn.functional as F

from plot_scripts.plots import plot_metrics
from preprocess import mnist_test_loader, mnist_train_loader, classes

from train_and_test import train, test, device

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3) # salida: 8x26x26 sin padding

        # Ojo, depende de cada modelo
        self.fc1 = nn.Linear(8 * 26  * 26, 120)
        self.fc2 = nn.Linear(120, 10)
       
    
    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out

net0 = Net() 
net0.to(device)

path, train_losses, train_accuracies, test_losses, test_accuracies = train(net0, mnist_train_loader, mnist_test_loader,path = ".mnist_net0.pth" )
test(net0, mnist_test_loader, classes, path)

plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, prefix="figures/mnist_m0_")