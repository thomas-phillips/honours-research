import torch
import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 7 * 9, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        # print("Shape after conv1:", x.shape)
        x = self.conv2(x)
        # print("Shape after conv2:", x.shape)
        x = self.conv3(x)
        # print("Shape after conv3:", x.shape)
        x = self.conv4(x)
        # print("Shape after conv4:", x.shape)
        x = self.flatten(x)
        # print("Shape after flattening:", x.shape)
        # print("Shape of the weight matrix:", self.linear.weight.shape)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions


class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(pretrained=True)

        # Modify the first convolutional layer to accept 1 input channel
        # Original input channels: 3
        # New input channels: 1
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the output layer to have the desired number of classes
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, num_classes)

        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)
