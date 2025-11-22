# Red neuronal con 2 capas intermedias (500,300) y una capa de salida SOFTMAX usando Cross Entropy como función de pérdida. Tasa de aprendizaje 0,01.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from plot_scripts.plots import plot_metrics, plot_confusion_matrix
from preprocess import mnist_test_loader, mnist_train_loader, classes

from train_and_test import train, test, device, get_all_preds_and_labels

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Usando dispositivo:", device)

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        # MNIST: imágenes 1x28x28 = 784 características
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 10)  # 10 clases (0–9)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = self.flatten(x)          # (batch, 784)
        x = F.relu(self.fc1(x))      # capa oculta 1
        x = F.relu(self.fc2(x))      # capa oculta 2
        logits = self.fc3(x)         # salida (logits)
        return logits

# Crear el modelo, entrenarlo y evaluarlo
model_0 = Mlp()
model_0.to(device)

path, train_losses, train_accuracies, test_losses, test_accuracies = train(
    model_0,
    mnist_train_loader,
    mnist_test_loader,
    learning_rate=0.01,
    epochs=5,
    optimizer_="SGD",            
    path="./mnist_mlp.pth"
)

test(model_0, mnist_test_loader, classes, path)

plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies, prefix=f"figures/mnist_mlp")

preds, labels = get_all_preds_and_labels(model_0, mnist_test_loader, device)
plot_confusion_matrix(preds, labels, classes, normalize=True, prefix = f"figures/confusion_matrix_mlp")