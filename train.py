import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import torchvision
import torchvision.transforms
import numpy as np 
import matplotlib.pyplot as plt

from ml.plotting import imshowPytorch
from ml.model import NeuralNet, LeNet


transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train = torchvision.datasets.FashionMNIST(root='./data/',
                                             train=True, 
                                             transform=transforms,
                                             download=True)
test = torchvision.datasets.FashionMNIST(root='.data/',
                                             train=False, 
                                             transform=transforms,
                                             download=True)

train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=8, 
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test,
                                           batch_size=8, 
                                           shuffle=False)
                                           
data_iter = iter(train_loader)
images, label = data_iter.next()
imshowPytorch(torchvision.utils.make_grid(images[0]))

##visualization
model = LeNet()
#model = NeuralNet(10)

module = model.conv1
print(list(module.named_parameters()))
print("before pruning: ",list(module.named_buffers()))
prune.random_unstructured(module, name="weight", amount=0.3)
print("after pruning ", list(module.named_parameters()))

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())
print("model ", model)
##training
for e in range(2):
    # define the loss value after the epoch
    losss = 0.0
    number_of_sub_epoch = 0
    # loop for every training batch (one epoch)
    for images, labels in train_loader:
        out = model(images)
        loss = criterion(out, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        losss += loss.item()
        number_of_sub_epoch += 1
    print("Epoch {}: Loss: {}".format(e, losss / number_of_sub_epoch))


### evaluate
correct = 0
total = 0
model.eval()
for images, labels in test_loader:
    outputs = model(images)
    print("outputs ", outputs)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('Test Accuracy of the model on the {} test images: {}% with PyTorch'.format(total, 100 * correct // total))
