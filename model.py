import torch
import torch.nn as nn
from torch.autograd import Variable

'''

FacialNetwork is a CNN that takes in an image and outputs a tensor
of facial keypoint locations.

Input images are going to be set to 224x224.

Output is 68 facial keypoints w/ x,y coordinates, so 1x(68*2), or
an output size of 1x136

The model is 4 convolutional blocks, consisting of:

1. Convolutional Layer into...
2. A ReLU activation layer into...
3. A max pooling layer into...
4. A dropout layer for training.

...and linear blocks, which consist of:

1. A fully connceted layer into...
2. A ReLU activation layer into...
3. A dropout layer when training.

The network architecture is thus:

1. Convolutional Block - In: (1, 224, 224) - Out: (32, 110, 110)
2. Convolutional Block - In: (32, 110, 110) - Out: (64, 53, 53)
3. Convolutional Block - In: (64, 53, 53) - Out: (128, 24, 24)
4. Convolutional Block - In: (128, 24, 24) - Out: (256, 10, 10)

5. Flatten layer - In: (256, 10, 10) - Out: (1, 25600)

6. Linear Block - In: (1, 25600) - Out: (1, 6400)
7. Linear Block - In: (1, 6400) - Out: (1, 1600)
8. Linear Block - In: (1, 1600) - Out: (1, 400)

9. A dense layer, our output - In: (1, 400) - Out: (1, 136)

'''

class FacialNetwork(nn.Module):

    def __init__(self):
        # Start with calling the super for built in functionalit
        super(FacialNetwork, self).__init__()

        # Here we'll actually build out the net. See above
        # for the architectural map, but I'll try and make it
        # clear here as well.

        # Create the four convolutional blocks:

        # 1
        # Input is 1x244x244 (pytorch is color channel first)
        # since we're grayscale, it's 1 channel
        # We're going to go with an out_channel (feature maps)
        # of 32 filters. I plan on just doubling from here
        # on each convolutional layer.
        conv = nn.Conv2d(in_channels = 1, out_channels = 32,
                kernel_size = 5, stride = 1, padding = 0)
        relu = nn.F.ReLU(conv)
        pool = nn.MaxPool2d(kernel_size = 2, stride = 2)(relu)
        dropout = nn.Dropout(p = 0.5)(pool)
        
        self.conv_block_1 = dropout
        # conv_block_1's output is
        # ( (224 - 5 + 2*0) / 1 ) + 1 = 220 for CNN then pool:
        # ( (220 - 2 + 2*0) / 2 ) + 1 = 110
        # Expected output of this is thus (32, 110, 110)

        # 2
        conv = nn.Conv2d(in_channels = 32, out_channels = 64,
                kernel_size = 5, stride = 1, padding = 0)
        relu = nn.F.ReLU(conv)
        pool = nn.MaxPool2d(kernel_size = 2, stride = 2)(relu)
        dropout = nn.Dropout(p = 0.5)(pool)

        self.conv_block_2 = dropout
        # conv_block_2's output is
        # (110 - 5) + 1 = 106 => ((106 - 2) / 2) + 1 = 53
        # Expected output of this is thus (64, 53, 53)

        # 3
        conv = nn.Conv2d(in_channels = 64, out_channels = 128,
                kernel_size = 5, stride = 1, padding = 0)
        relu = nn.F.ReLU(conv)
        pool = nn.MaxPool2d(kernel_size = 2, stride = 2)(relu)
        dropout = nn.Dropout(p = 0.5)(pool)

        self.conv_block_3 = dropout
        # conv_block_3's output is
        # (53 - 5) + 1 = 49 => ((49 - 2) / 2) + 1 = 24.5 => 24
        # note that default maxpooling in torch floors the #
        # Expeted output of this is thus (128, 24, 24)

        # 4
        conv = nn.Conv2d(in_channels = 128, out_channels = 256,
                kernel_size = 5, stride = 1, padding = 0)
        relu = nn.F.ReLU(conv)
        pool = nn.MaxPool2d(kernel_size = 2, stride = 2)(relu)
        dropout = nn.Dropout(p = 0.5)(pool)

        self.conv_block_4 = dropout
        # conv_block_4's output is
        # (24 -5) + 1 = 20 => ((20 - 2) / 2) + 1 = 10
        # Expected output is (256, 10, 10)


        # Create the linear blocks

        # 1
        linear = nn.Linear(in_features = 25600, out_features = 6400)
        relu = nn.F.ReLU(linear)
        dropout = nn.Dropout(p = 0.5)
        
        self.linear_block_1 = dropout
        
        # 2
        linear = nn.Linear(in_features = 6400, out_features = 1600)
        relu = nn.F.ReLU(linear)
        dropout = nn.Dropout(p = 0.5)
        
        self.linear_block_2 = dropout

        # 3
        linear = nn.Linear(in_features = 6400, out_features = 400)
        relu = nn.F.ReLU(linear)
        dropout = nn.Dropout(p = 0.5)
        
        self.linear_block_3 = dropout


        # Finally, the dense output layer
        self.output_layer = nn.Linear(in_features = 400, out_features = 136)


    def forward(self, x):
        
        # Convolutional blocks
        x = self.conv_block_1(x)
        print(x.shape)

        x = self.conv_block_2(x)
        print(x.shape)

        x = self.conv_block_3(x)
        print(x.shape)

        x = self.conv_block4(x)
        print(x.shape)

        # Flatten
        x = x.view(x.size(0), -1)
        print(x.shape)

        # Linear blocks
        x = self.linear_block_1(x)
        print(x.shape)

        x = self.linear_block_2(x)
        print(x.shape)

        x = self.linear_block_3(x)
        print(x.shape)

        # Final dense layer output
        x = self.output_layer(x)
        print(x.shape)

        return x




