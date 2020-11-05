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


model = LeNetPL()
print("model ", model)
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
            
for i in range(10):
    print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))
