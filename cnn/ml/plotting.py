import numpy as np 
import matplotlib.pyplot as plt

def imshowPytorch(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('images/test.png')
