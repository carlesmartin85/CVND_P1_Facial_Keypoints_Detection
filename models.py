## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def weights_init(mods):
    if type(mods) == nn.Conv2d:
        I.uniform_(mods.weight, a=-0.1, b=0.1)
    if type(mods) == nn.Linear:
        I.uniform_(mods.weight, a=-0.1, b=0.1)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2) # Output Size: **reduces to half size of height and width each time it is called**

        # first conv layer: 1 input image channel (grayscale), 5 output channels/feature maps, 5x5 square convolution kernel
        # output size = (W - F)/S + 1 = (224 - 5)/1 + 1 =220
        self.conv1 = nn.Conv2d(1, 5, 5) # Output Size: (5, 220, 220)
        # after another pool layer this becomes (5, 110, 110)

        # second conv layer: 5 inputs, 10 outputs, 5x5 conv
        # output size = (W - F)/S + 1 = (110 - 5)/1 + 1 =106
        self.conv2 = nn.Conv2d(5, 10, 5) # Output Size: (10, 106, 106)
        # after another pool layer this becomes (10, 53, 53)

        # third conv layer: 10 inputs, 20 outputs, 4x4 conv
        # output size = (W - F)/S + 1 = (53 - 4)/1 + 1 = 50
        self.conv3 = nn.Conv2d(10, 20, 4)  # Output Size: (20, 50, 50)
        # after another pool layer this becomes (20, 25, 25)

        # fourth conv layer: 20 inputs, 40 outputs, 4x4 conv
        # output size = (W - F)/S + 1 = (25 - 4)/1 + 1 = 22
        self.conv4 = nn.Conv2d(20, 40, 4)  # Output Size: (40, 22, 22)
        # after another pool layer this becomes (40, 11, 11)

        # fifth conv layer: 40 inputs, 80 outputs, 2x2 conv
        # output size = (W - F)/S + 1 = (11 - 2)/1 + 1 = 10
        self.conv5 = nn.Conv2d(40, 80, 2)  # Output Size: (80, 10, 10)
        # after another pool layer this becomes (80, 5, 5)

        # dropout with p = 0.1 - 0.2
        self.drop_conv = nn.Dropout(p=0.1)
        self.drop_fc = nn.Dropout(p=0.2)

        # finally, create 68*2 channels, 2 for each of the 68 keypoint (x, y) pairs
        self.fc1 = nn.Linear(80*5*5, 68*8)
        self.fc2 = nn.Linear(68*8, 68*4)
        self.fc3 = nn.Linear(68*4, 68*2)

        # batchnorm2d layers
        self.bn1 = nn.BatchNorm2d(5)
        self.bn2 = nn.BatchNorm2d(10)
        self.bn3 = nn.BatchNorm2d(20)
        self.bn4 = nn.BatchNorm2d(40)
        self.bn5 = nn.BatchNorm2d(80)

        # init-ing net weigths
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv3.apply(weights_init)
        self.conv4.apply(weights_init)
        self.conv5.apply(weights_init)
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)
        self.fc3.apply(weights_init)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:

        x = self.drop_conv(self.pool(self.bn1(F.relu(self.conv1(x)))))
        x = self.drop_conv(self.pool(self.bn2(F.relu(self.conv2(x)))))
        x = self.drop_conv(self.pool(self.bn3(F.relu(self.conv3(x)))))
        x = self.drop_conv(self.pool(self.bn4(F.relu(self.conv4(x)))))
        x = self.drop_conv(self.pool(self.bn5(F.relu(self.conv5(x)))))

        x = x.view(x.shape[0], -1)
        x = self.drop_fc(self.fc1(x))
        x = self.drop_fc(self.fc2(x))
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x