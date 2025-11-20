# Una red convolucional con la siguiente configuración: INPUT -> CONV -> RELU -> FC -> RELU -> FC

import torch

import torch.nn as nn
import torch.nn.functional as F

from plot_scripts.plots import plot_metrics, plot_confusion_matrix
from preprocess import mnist_test_loader, mnist_train_loader, classes

from train_and_test import train, test, device, get_all_preds_and_labels

class Net(nn.Module):

    def __init__(self, in_channels_, out_channels_, kernel_size_, padding_, hidden_layer_, stride_ = 1, input_size_=28):

        super().__init__()

        self.conv1 = nn.Conv2d(in_channels_, out_channels_, kernel_size_, padding=padding_) 

        output_dim = (input_size_ - kernel_size_ + 2 * padding_) // stride_ + 1

        self.fc1 = nn.Linear(out_channels_ * output_dim * output_dim, hidden_layer_)
        self.fc2 = nn.Linear(hidden_layer_, 10)
       
    
    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out

# Versiones para el modelo 1

net1_v0 = Net(in_channels_=1, out_channels_=8, kernel_size_=3, padding_=0, hidden_layer_=64)

net1_v1 = Net(in_channels_=1, out_channels_=8, kernel_size_=3, padding_=0, hidden_layer_=128)

net1_v2 = Net(in_channels_=1, out_channels_=8, kernel_size_=3, padding_=1, hidden_layer_=128) 

net1_v3 = Net(in_channels_=1, out_channels_=8, kernel_size_=5, padding_=1, hidden_layer_=128) 

models = [net1_v0, net1_v1, net1_v2, net1_v3]

# Usar la gpu si es el caso
if device == torch.device("cuda:0"):
    for model in models:
        model.to(device)


# Entrenamiento, preubas y metricas para los modelos
for i in range(0, len(models)):

    print(f"ENTRENAMIENTO PARA EL MODELO 1 VERSION {i}")
    path, train_losses, train_accuracies, test_losses, test_accuracies = train(models[i], mnist_train_loader, mnist_test_loader, path = f".mnist_net1_v{i}.pth" )
    print()
    
    print(f"TEST PARA CADA CLASE DEL MODELO 1 VERSION {i}")
    test(models[i], mnist_test_loader, classes, path)
    print()

    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, prefix=f"figures/mnist_net1_v{i}")

    preds, labels = get_all_preds_and_labels(models[i], mnist_test_loader, device)
    plot_confusion_matrix(preds, labels, classes, normalize=True, prefix = f"figures/confusion_matrix_net1_v{i}")
