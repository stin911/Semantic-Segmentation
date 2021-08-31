import numpy as np
import matplotlib.pyplot as plt


def plot(array: np.array, grey=False):
    array = np.transpose(array, (1, 2, 0))

    #array = (array > 0.2).astype(np.uint8)
    if not grey:

        plt.imshow(array, cmap="Greys")
        plt.show()
    else:

        plt.imshow(array)
        plt.show()
