import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from transforms import RescaleImage, NormalizeImage, ToTensorImage
import math
import numpy as np

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
        
        self.image_shape = (224, 224)
        self.build_network()
        self.prepare_for_inference()

        
    def build_network(self):
        # Here we'll create most of the layers, and construct it
        # within the forward() function. I'll try and keep clear
        # the sizes of convolutional / linear layers

        # First, we'll need a pooling and dropout layer that
        # will be reused
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.dropout = nn.Dropout(p = 0.5)

        # Now create the four convolutional layers.
        # Where I mention _block, it is including pooling in its
        # output size calculation

        # 1
        # Input is 1x244x244 (pytorch is color channel first)
        # since we're grayscale, it's 1 channel
        # We're going to go with an out_channel (feature maps)
        # of 32 filters. I plan on just doubling from here
        # on each convolutional layer.
        self.conv_1 = nn.Conv2d(in_channels = 1, out_channels = 32,
                kernel_size = 5, stride = 1, padding = 0)
        
        # conv_block_1's output is
        # ( (224 - 5 + 2*0) / 1 ) + 1 = 220 for CNN then pool:
        # ( (220 - 2 + 2*0) / 2 ) + 1 = 110
        # Expected output of this is thus (32, 110, 110)

        # 2
        self.conv_2 = nn.Conv2d(in_channels = 32, out_channels = 64,
                kernel_size = 5, stride = 1, padding = 0)

        # conv_block_2's output is
        # (110 - 5) + 1 = 106 => ((106 - 2) / 2) + 1 = 53
        # Expected output of this is thus (64, 53, 53)

        # 3
        self.conv_3 = nn.Conv2d(in_channels = 64, out_channels = 128,
                kernel_size = 5, stride = 1, padding = 0)

        # conv_block_3's output is
        # (53 - 5) + 1 = 49 => ((49 - 2) / 2) + 1 = 24.5 => 24
        # note that default maxpooling in torch floors the #
        # Expeted output of this is thus (128, 24, 24)

        # 4
        self.conv_4 = nn.Conv2d(in_channels = 128, out_channels = 256,
                kernel_size = 5, stride = 1, padding = 0)

        # conv_block_4's output is
        # (24 -5) + 1 = 20 => ((20 - 2) / 2) + 1 = 10
        # Expected output is (256, 10, 10)


        # Create the linear blocks

        # 1
        self.linear_1 = nn.Linear(in_features = 25600, out_features = 6400)
        
        # 2
        self.linear_2 = nn.Linear(in_features = 6400, out_features = 1600)
        
        # 3
        self.linear_3 = nn.Linear(in_features = 1600, out_features = 400)

        # Finally, the dense output layer
        self.output_layer = nn.Linear(in_features = 400, out_features = 136)

    def prepare_for_inference(self):
        rescale = RescaleImage(self.image_shape)
        normalize = NormalizeImage()
        tensor = ToTensorImage()
        self.transformer = transforms.Compose([rescale, normalize, tensor])

    def forward(self, x):
        
        # Convolutional blocks
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv_2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv_3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = self.conv_4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Linear blocks
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.linear_3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Final dense layer output
        x = self.output_layer(x)

        return x

    def inference(self, image):
        # Copy the image to prevent transforming the original
        image = np.copy(image)
        image = self.transformer(image)
        images = image.view(1, 1, self.image_shape[0], self.image_shape[1])

        # Check for CUDA
        if torch.cuda.is_available():
            images = images.type(torch.cuda.FloatTensor)
        else:
            images = images.type(torch.FloatTensor)

        # Forward pass
        output = self(images)

        # Generate the keypoints from the give nvalues
        coords = []
        coord = None
        output = output[0].detach().numpy()

        for index, value in enumerate(list(output)):
            # Denormalize!
            value = math.floor((value * 50) + 100)

            if index % 2 == 0 :
                coord = value
            else :
                coord = [coord, value]
                coords.append(coord)
                coord = None
        
        return coords