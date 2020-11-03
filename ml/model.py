import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, num_of_class):
        super(NeuralNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),
            ##in_channels (1) – Number of channels in the input image
            ##out_channels (6) – Number of channels produced by the convolution
            ##kernel_size (5) – Size of the convolving kernel
            ##stride (1) – Stride of the convolution. Default: 1
            ##padding (2) – Zero-padding added to both sides of the input. Default: 0
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            ##kernel_size (2) – the size of the window
            ##stride (2) – the stride of the window. Default value is kernel_size
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
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
