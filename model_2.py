#------------------     PROPOSITO DE ESTE ARCHIVO   ------------------#
# Una red convolucional con una configuración seleccionada (y justificada) por el equipo
# evaluar varios modelos y escoger el mejor entre ellos!!!!

import torch

import torch.nn as nn
import torch.nn.functional as F

from plot_scripts.plots import  plot_metrics, plot_confusion_matrix
from preprocess import mnist_test_loader, mnist_train_loader, classes
from train_and_test import train, test, get_all_preds_and_labels,  device



#------------------     CNN VERSION 0    ------------------#
class Net0(nn.Module):

    def __init__(self, in_channels_, out_channels_, kernel_size_, padding_, hidden_layer_, stride_ = 1, input_size_=28):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels_, out_channels_, kernel_size_, padding=padding_) 

        self.pool = nn.MaxPool2d(2, 2)                         

        output_dim = (input_size_ - kernel_size_ + 2 * padding_) // stride_ + 1

        self.fc1 = nn.Linear(out_channels_ * output_dim // 2 * output_dim // 2, hidden_layer_)
        self.fc2 = nn.Linear(hidden_layer_, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


#------------------     CNN VERSION PAPER    ------------------#

class NetPaper(nn.Module):

    def __init__(self, in_channels_, out_channels_1, out_channels_2, out_channels_3, out_channels_4, kernel_size_, padding_, hidden_layer_, stride_ = 1, input_size_=28):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels_, out_channels_1, kernel_size_, padding=padding_) 


        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size_, padding=padding_) 


        self.pool1 = nn.MaxPool2d(2, 2)  

        self.dropout_conv1 = nn.Dropout(p=0.25)   


        self.conv3 = nn.Conv2d(out_channels_2, out_channels_3, kernel_size_, padding=padding_) 

        # Dropout

        self.conv4 = nn.Conv2d(out_channels_3, out_channels_4, kernel_size_, padding=padding_) 

        self.pool2 = nn.MaxPool2d(2, 2)  

        self.dropout_conv2 = nn.Dropout(p=0.25)   


        # dropout
        output_dim = (input_size_ - kernel_size_ + 2 * padding_) // stride_ + 1

        self.fc1 = nn.Linear(out_channels_4 *  output_dim // 4 *  output_dim // 4, hidden_layer_)

        self.dropout_fc   = nn.Dropout(p=0.5)

        # dropout

        self.fc2 = nn.Linear(hidden_layer_, 10)
    
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.dropout_conv1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = self.dropout_conv2(x)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x


#------------------     CNN VERSION 1    ------------------#
class Net1(nn.Module):

    def __init__(self, in_channels_, out_channels_1, out_channels_2, out_channels_3, out_channels_4, kernel_size_, padding_, hidden_layer_, stride_ = 1, input_size_=28):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels_, out_channels_1, kernel_size_, padding=padding_) 


        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size_, padding=padding_) 


        self.pool1 = nn.MaxPool2d(2, 2)  


        self.conv3 = nn.Conv2d(out_channels_2, out_channels_3, kernel_size_, padding=padding_) 

        # Dropout

        self.conv4 = nn.Conv2d(out_channels_3, out_channels_4, kernel_size_, padding=padding_) 

        self.pool2 = nn.MaxPool2d(2, 2)  

        # dropout
        output_dim = (input_size_ - kernel_size_ + 2 * padding_) // stride_ + 1

        self.fc1 = nn.Linear(out_channels_4 *  output_dim // 4 *  output_dim // 4, hidden_layer_)

        # dropout

        self.fc2 = nn.Linear(hidden_layer_, 10)
    
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

#------------------     CNN VERSION 2    ------------------#

class Net2(nn.Module):

    def __init__(self, in_channels_, out_channels_1, out_channels_2, kernel_size_, padding_, hidden_layer_, stride_ = 1, input_size_=28):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels_, out_channels_1, kernel_size_, padding=padding_) 

        self.pool1 = nn.MaxPool2d(2, 2)  


        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size_, padding=padding_) 

        # Dropout

        self.pool2 = nn.MaxPool2d(2, 2)  

        # dropout
        output_dim = (input_size_ - kernel_size_ + 2 * padding_) // stride_ + 1

        self.fc1 = nn.Linear(out_channels_2 * (output_dim // 4) * (output_dim // 4), hidden_layer_)

        # dropout

        self.fc2 = nn.Linear(hidden_layer_, 10)
    
    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



#------------------     CNN VERSION 3    ------------------#

class Net3(nn.Module):

    def __init__(self, in_channels_, out_channels_1, out_channels_2, kernel_size_, padding_, hidden_layer_, stride_ = 1, input_size_=28):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels_, out_channels_1, kernel_size_, padding=padding_) 

        self.pool1 = nn.MaxPool2d(2, 2)  


        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size_, padding=padding_) 

        # Dropout

        self.pool2 = nn.MaxPool2d(2, 2)  

        # dropout

        self.dropout_conv = nn.Dropout(p=0.25)   

        output_dim = (input_size_ - kernel_size_ + 2 * padding_) // stride_ + 1

        self.fc1 = nn.Linear(out_channels_2 * (output_dim // 4) * (output_dim // 4), hidden_layer_)

        self.dropout_fc   = nn.Dropout(p=0.5)

        # dropout

        self.fc2 = nn.Linear(hidden_layer_, 10)
    
    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x
    



#------------------    INSTANCIAS DE CADA DE CNN  ------------------#

net2_v0 = Net0(1, 32, 3, 1, 128)
net2_paper = NetPaper(1, 32, 32, 64, 64, 3, 1, 512)
net2_v1 = Net1(1, 32, 32, 64, 64, 3, 1, 512) 
net2_v2 = Net2(1, 32, 64, 3, 1, 512) 
net2_v3 = Net3(1, 32, 64, 3, 1, 512) 

models_m2 = [net2_v0, net2_v1, net2_v2, net2_v3]

# Usar la gpu si es el caso
if device == torch.device("cuda:0"):
    net2_paper.to(device)
    for model in models_m2:
        model.to(device)

print(f"ENTRENAMIENTO PARA EL MODELO BASADO EN EL PAPER")
path, train_losses, train_accuracies, test_losses, test_accuracies = train(net2_paper, mnist_train_loader, mnist_test_loader, path = f".mnist_net2_paper.pth", optimizer_="Adam" )
print()

print(f"TEST PARA CADA CLASE DEL MODELO BASADO EN EL PAPER")
test(net2_paper, mnist_test_loader, classes, path)
print()
plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, prefix=f"figures/mnist_net2_paper")
preds, labels = get_all_preds_and_labels(net2_paper, mnist_test_loader, device)
plot_confusion_matrix(preds, labels, classes, normalize=True, prefix = f"figures/confusion_matrix_net2_paper")


# Entrenamiento, preubas y metricas para los modelos
for i in range(0, len(models_m2)):

    print(f"ENTRENAMIENTO PARA EL MODELO 2 VERSION {i}")
    path, train_losses, train_accuracies, test_losses, test_accuracies = train(models_m2[i], mnist_train_loader, mnist_test_loader, path = f".mnist_net2_v{i}.pth" )
    print()
    
    print(f"TEST PARA CADA CLASE DEL MODELO 2 VERSION {i}")
    test(models_m2[i], mnist_test_loader, classes, path)
    print()

    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, prefix=f"figures/mnist_net2_v{i}")

    preds, labels = get_all_preds_and_labels(models_m2[i], mnist_test_loader, device)
    plot_confusion_matrix(preds, labels, classes, normalize=True, prefix = f"figures/confusion_matrix_net2_v{i}")
