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
trainer = Trainer(max_epochs=1, profiler="simple")
trainer.fit(model)
