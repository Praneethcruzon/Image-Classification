import torch.nn as nn
from collections import OrderedDict


class CNN(nn.Module):
    # A 10 Layer Deep Convolution Neural Network for Binary Image Classification.
    def __init__(self, in_channels = 3, out_channels = 1):
        super().__init__()
        
        # Input Channels 
        self.in_channels = in_channels
        # Output Channels defaults to 1 as its a Binary Classification model. 
        self.out_channels = out_channels

        # Initial Number of Features
        self.features = 4

        # Initializing 10 layers of convolution
        self.layer_1 = self._block(in_channels = self.in_channels, out_channels = self.features, name = "Layer 1")
        self.layer_2 = self._block(in_channels = self.features, out_channels = self.features * 2, name = "Layer 2")
        self.layer_3 = self._block(in_channels = self.features * 2, out_channels = self.features * 4, name = "Layer 3")
        self.layer_4 = self._block(in_channels = self.features * 4, out_channels = self.features * 8, name = "Layer 4")
        self.layer_5 = self._block(in_channels = self.features * 8, out_channels = self.features * 16, name = "Layer 5")
        self.layer_6 = self._block(in_channels = self.features * 16, out_channels = self.features * 32, name = "Layer 6")
        self.layer_7 = self._block(in_channels = self.features * 32, out_channels = self.features * 64, name = "Layer 7")
        self.layer_8 = self._block(in_channels = self.features * 64, out_channels = self.features * 128, name = "Layer 8")
        self.layer_9 = self._block(in_channels = self.features * 128, out_channels = self.features * 256, name = "Layer 9")
        self.layer_10 = self._block(in_channels = self.features * 256, out_channels = self.features * 512, name = "Layer 10")
        # Max Pooling Layer
        self.max_pooling = nn.MaxPool2d(kernel_size = 3, stride = 1)
        # Flatten Layer
        self.flatten = nn.Flatten()
        # Fully Connected Layers
        self.linear_1 = nn.Linear(in_features = self.features * 512, out_features = self.features * 128)
        self.linear_2 = nn.Linear(in_features = self.features * 128, out_features = self.features * 32)
        self.linear_3 = nn.Linear(in_features = self.features * 32, out_features = self.out_channels)

    def forward(self, x):

        x = self.max_pooling(self.layer_1(x))
        x = self.max_pooling(self.layer_2(x)) 
        x = self.max_pooling(self.layer_3(x)) 
        x = self.max_pooling(self.layer_4(x)) 
        x = self.max_pooling(self.layer_5(x)) 
        x = self.max_pooling(self.layer_6(x)) 
        x = self.max_pooling(self.layer_7(x)) 
        x = self.max_pooling(self.layer_8(x)) 
        x = self.max_pooling(self.layer_9(x)) 
        x = self.max_pooling(self.layer_10(x)) 
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)

        return x

    def _block(self, in_channels, out_channels, name):
        # A Template representing the structure of a block.
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + " Convolution",
                        nn.Con2d(
                            in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = 3
                        )
                    ),
                    (
                        name + " Batch Normalization",
                        nn.BatchNorm2d(num_features = out_channels)
                    ),
                    (
                        name + " ReLU Activation",
                        nn.ReLU()
                    )
                ]
            )
        )
