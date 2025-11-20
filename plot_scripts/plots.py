import matplotlib.pyplot as plt
import numpy as np


from sklearn.metrics import confusion_matrix
import seaborn as sns

# functions to show an image

def imshow(img):
    # Hay que chequear esta linea
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("plot.png")
    plt.show()


def plot_metrics(train_losses, test_losses, train_accs, test_accs, prefix="mnist"):
    epochs = range(1, len(train_losses) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, test_losses, label="Test Loss", marker='x')
    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_loss.png")
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Accuracy", marker='o')
    plt.plot(epochs, test_accs, label="Test Accuracy", marker='x')
    plt.title("Accuracy per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{prefix}_accuracy.png")
    plt.close()


def plot_confusion_matrix(preds, labels, classes, normalize=False, prefix="confusion_matrix" ):

    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d",
                xticklabels=classes, yticklabels=classes, cmap="Blues")
    
    plt.xlabel("Prediction")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    plt.tight_layout()
    plt.savefig(f"{prefix}.png")
    plt.close()