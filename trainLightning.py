import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from pytorch_lightning import Trainer


import torchvision
import torchvision.transforms

import numpy as np 
import matplotlib.pyplot as plt

from ml.plotting import imshowPytorch, output_label
from ml.model import NeuralNet, LeNetPL

import pytorch_lightning as pl

frac = 0.1
model = LeNetPL()
print("model ", model)
module = model.conv1
print("module.named_parameters(): ", list(module.named_parameters()))
## named parameters of the conv1 layer are 
    #'weight' which is a tensor of weights initialized randomly of n * kernel_size*kernel_size where n is the number of output channels
    #'bias' which is a 1 D tensor of length of the output channels
print("before pruning: ",list(module.named_buffers()))
prune.random_unstructured(module, name="weight", amount=frac)
# the "amount" specifies the amount of weights that are pruned away, i.e. set to zero
print("after pruning: ", list(module.named_parameters()))
#after pruning the named_parameters will show 'weight_orig' instead of 'weight'
print("module.named_buffers(): ",list(module.named_buffers()))
#after pruning the named_buffers shows the 'weight_mask'
print("module.weight: ", module.weight)
print("module._forward_pre_hooks: ",module._forward_pre_hooks)

trainer = Trainer(max_epochs=10, profiler="simple")
trainer.fit(model)




### evaluate
correct = 0
total = 0
model.eval()
test_loader = model.test_dataloader()
for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the {} test images: {}% with PyTorch'.format(total, 100 * correct // total))
class_correct = [0. for _ in range(10)]
total_correct = [0. for _ in range(10)]

with torch.no_grad():
    for images, labels in test_loader:
        test = Variable(images)
        outputs = model(test)
        predicted = torch.max(outputs, 1)[1]
        c = (predicted == labels).squeeze()
        
        for i in range(15):
            label = labels[i]
            class_correct[label] += c[i].item()
            total_correct[label] += 1
            
with open('pruning_'+str(frac)+'.txt', 'w') as f:
    f.write('Test Accuracy of the model on the {} test images: {}% with PyTorch'.format(total, 100 * correct // total))
    for i in range(10):
        print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))
        f.write("Accuracy of {}: {:.2f}% \n".format(output_label(i), class_correct[i] * 100 / total_correct[i]))
