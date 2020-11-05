import torch
import numpy as np 
import matplotlib.pyplot as plt

def imshowPytorch(img, i):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('images/images_'+str(i)+'.png')
def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]

def plotLoss(iteration_list, loss_list):
    plt.plot(iteration_list, loss_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Loss")
    plt.title("Iterations vs Loss")
    plt.savefig('images/loss.png')
