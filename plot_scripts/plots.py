import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    # Hay que chequear esta linea
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig("plot.png")
    plt.show()


