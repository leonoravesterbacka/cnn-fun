import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.autograd import Variable


import torchvision
import torchvision.transforms

import itertools
import numpy as np 
import matplotlib.pyplot as plt

from ml.plotting import imshowPytorch, plotLoss, output_label
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
                                           batch_size=32, 
                                           shuffle=False, 
                                           num_workers=10)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                           batch_size=32, 
                                           shuffle=False,
                                           num_workers=10)

labelNames = ['tshirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'boot' ]
#for images, labels in test_loader:
#    print("label ", labelNames[labels])
#    imshowPytorch(torchvision.utils.make_grid(images), labels)

data_iter = iter(train_loader)
images, label = data_iter.next()
##visualization
model = LeNet()
#model = NeuralNet(10)

module = model.conv1
print("module.named_parameters(): ", list(module.named_parameters()))
## named parameters of the conv1 layer are 
    #'weight' which is a tensor of weights initialized randomly of n * kernel_size*kernel_size where n is the number of output channels
    #'bias' which is a 1 D tensor of length of the output channels
print("before pruning: ",list(module.named_buffers()))
prune.random_unstructured(module, name="weight", amount=0.3)
# the "amount" specifies the amount of weights that are pruned away, i.e. set to zero
print("after pruning: ", list(module.named_parameters()))
#after pruning the named_parameters will show 'weight_orig' instead of 'weight'
print("module.named_buffers(): ",list(module.named_buffers()))
#after pruning the named_buffers shows the 'weight_mask'
print("module.weight: ", module.weight)
print("module._forward_pre_hooks: ",module._forward_pre_hooks)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters())
print("model ", model)
##training
count = 0
loss_list = []
iteration_list = []
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
        count += 1

    loss_list.append(loss.item())
    iteration_list.append(count)

    print("Epoch {}: Loss: {}".format(e, losss / number_of_sub_epoch))

plotLoss(iteration_list, loss_list)
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
class_correct = [0. for _ in range(10)]
total_correct = [0. for _ in range(10)]

with torch.no_grad():
    for images, labels in test_loader:
        #images, labels = images.to(device), labels.to(device)
        test = Variable(images)
        outputs = model(test)
        predicted = torch.max(outputs, 1)[1]
        c = (predicted == labels).squeeze()
        
        for i in range(15):
            label = labels[i]
            class_correct[label] += c[i].item()
            total_correct[label] += 1
        
for i in range(10):
    print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))
