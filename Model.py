import torch.nn as nn


class ModelQD(nn.Module) :
    def __init__(self,input_size = 28, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        # self.fc3 = nn.Linear(128, num_classes)
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # x = self.fc3(x)
        return x




        
