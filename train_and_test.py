
import torch

import torch.nn as nn
import torch.optim as optim # Funcion de perdida y optimizador


#------------------     TRABAJAR CON CPU O GPU    ------------------#
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


#------------------     ENTRENAMIENTO Y TEST    ------------------#

def train(net, trainloader, testloader, learning_rate=0.01, epochs = 5, path = "./mnist_net.pth"):

    # Funcion de perdida, especifica para problemas multiclase
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    # Listas para guardar el historial de perdidas y de precision para luego hacer graficas
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs): 

        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(100 * correct / total)

        # Evaluar el modelo actual en el conjunto test
        net.eval()
        test_loss, correct_test, total_test = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(testloader))
        test_accuracies.append(100 * correct_test / total_test)

        print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Test Loss={test_losses[-1]:.4f}, Train Acc={train_accuracies[-1]:.2f}%, Test Acc={test_accuracies[-1]:.2f}%")

    

    print('Finished Training')


    torch.save(net.state_dict(), path)
    print("Modelo guardado en:", path)

    return path, train_losses, train_accuracies, test_losses, test_accuracies


def test(net, testloader, classes, path):

   
    net.load_state_dict(torch.load(path, weights_only=True))

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

