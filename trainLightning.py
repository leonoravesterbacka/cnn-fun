import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer


import torchvision
import torchvision.transforms

import numpy as np 
import matplotlib.pyplot as plt

from ml.plotting import imshowPytorch
from ml.model import NeuralNet, LeNet

import pytorch_lightning as pl


model = LeNet()
print("model ", model)
trainer = Trainer(max_epochs=100, profiler="simple")
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
