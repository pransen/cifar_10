import torch.nn as nn
import torch.nn.functional as f


class NeuralNet(nn.Module):

    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(12, 24, 3, padding=1)
        self.conv4 = nn.Conv2d(24, 32, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(8 * 8 * 32, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = f.relu(self.conv2(f.relu(self.conv1(x))))
        x = self.pool2(x)
        x = f.relu(self.conv4(f.relu(self.conv3(x))))
        x = self.pool4(x)
        x = x.view(-1, 8 * 8 * 32)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x
