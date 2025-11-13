# Una red convolucional con la siguiente configuración: INPUT -> CONV -> RELU -> FC -> RELU -> FC
import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # Funcion de perdida y optimizador

from plot_scripts.plots import imshow
from preprocess import mnist_test_loader, mnist_train_loader, classes

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)

        # Ojo, depende de cada modelo
        # self.fc1 = nn.Linear(6 * 3 * 3, 120)
        # Revisar los canales de entrada
        self.fc1 = nn.Linear(4056, 120)
        self.fc2 = nn.Linear(120, 10)
       
    
    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return out


net = Net() 
net.to(device)


def train( trainloader, learning_rate=0.01, epochs = 10):

    # Funcion de perdida, especifica para problemas multiclase
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs): 

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './mnist_net.pth'
    torch.save(net.state_dict(), PATH)

    return PATH





def test(testloader, classes, PATH):

   
    net.load_state_dict(torch.load(PATH, weights_only=True))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


PATH = train(mnist_train_loader)
test(mnist_test_loader, classes, PATH)