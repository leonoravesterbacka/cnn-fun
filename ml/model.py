import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    def __init__(self, num_of_class):
        super(NeuralNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=2),
            ##in_channels (1) – Number of channels in the input image
            ##out_channels (6) – Number of channels produced by the convolution
            ##kernel_size (5) – Size of the convolving kernel
            ##stride (1) – Stride of the convolution. Default: 1. Stride is the amount you want to skip in a particular direction
            ##padding (2) – Zero-padding added to both sides of the input. Default: 0. padding is the additional boundary around the kernel
            ## great graphics on this: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ##kernel_size (2) – the size of the window
            ##stride (2) – the stride of the window. Default value is kernel_size
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.fc_model = nn.Sequential(
            nn.Linear(400,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(84, 10)

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(-1, 16*5*5)
        x = self.fc_model(x)
        x = self.classifier(x)
        return x


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 3)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

