import torch
import torchvision
from torchvision import datasets, transforms
from plot_scripts.plots import imshow

# Descargar dataset. Se guardan en ./data

batch_size = 128

# Conjunto de entrenamiento
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

mnist_train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)

# Conjunto de prueba
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=2)

classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

# Salida

# get some random training images
dataiter = iter(mnist_train_loader)
images, labels = next(dataiter)
