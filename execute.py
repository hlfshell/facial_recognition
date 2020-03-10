'''

exectue.py will load up a given model and image.

It perform inference with the model via the image,
and output the results

'''

import argparse
import transforms
from model import FacialNetwork
from torchvision import transforms as torchtransforms
from PIL import Image
import numpy as np
import cv2
import torch
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = "Execute a model on an image")

parser.add_argument("--image", help = "Image file to use", required = True)
parser.add_argument("--model", help = "Model file to load", required = True)

args = parser.parse_args()

# Prepare the transforms!
image_shape = (224, 224)
rescale = transforms.RescaleImage(image_shape)
normalize = transforms.NormalizeImage()
tensor = transforms.ToTensorImage()
transformer = torchtransforms.Compose([rescale, normalize, tensor])

# Create our model
model = FacialNetwork()
model.load_state_dict(torch.load(args.model))
model.eval() # Make sure we're just running inference

# Load our image and transform it
image = cv2.imread(args.image)
image_copy = np.copy(image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = tensor.image_transform(normalize.image_transform(rescale.image_transform(image)))
image = transformer(image)
images = image.view(1, 1, image_shape[0], image_shape[1])
image_copy = rescale(image_copy)
images = images.type(torch.FloatTensor)

# Generate our output
output = model(images)

# Pair the keypoints to be a set of 68 (x, y) coordinates
coords = []
x = []
y = []
coord = None
output = output[0].detach().numpy()

for index, value in enumerate(list(output)):
    # Denormalize!
    value = (value * 50) + 100

    if index % 2 == 0 :
        coord = value
        x.append(value)
    else :
        coord = [coord, value]
        coords.append(coord)
        coord = None
        y.append(value)

print(coords)

plt.figure()
plt.subplot(1, 2, 1).imshow(image_copy)
plt.subplot(1, 2, 2).imshow(image_copy)
plt.scatter(x, y, s=20, marker='.', c='m')
plt.savefig("result.png")