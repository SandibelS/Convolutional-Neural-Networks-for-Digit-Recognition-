#------------------     PROPOSITO DE ESTE ARCHIVO   ------------------#
# Una red convolucional con una configuración seleccionada (y justificada) por el equipo
# evaluar varios modelos y escoger el mejor entre ellos!!!!

import torch

import torch.nn as nn
import torch.nn.functional as F

from plot_scripts.plots import  plot_metrics
from preprocess import mnist_test_loader, mnist_train_loader, classes
from train_and_test import train, test, device


#------------------     CNN VERSION 1    ------------------#
class Net1(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3) # salida: 8x26x26 sin padding
        self.pool = nn.MaxPool2d(2, 2)              # salida: 8x13x13

        self.fc1 = nn.Linear(8 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


#------------------    INSTANCIAS DE CADA DE CNN  ------------------#

net1 = Net1() 
net1.to(device)

path, train_losses, train_accuracies, test_losses, test_accuracies = train(net1, mnist_train_loader, mnist_test_loader, path="./net/mnist_net1.pth")
test(net1, mnist_test_loader, classes, path)

# plot_training_metrics(train_losses, train_accuracies)
plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, prefix="figures/mnist_m2_v1_")