'''

exectue.py will load up a given model and image.

It perform inference with the model via the image,
and output the results

'''

import argparse
import transforms
from model import FacialNetwork
import numpy as np
import cv2
import torch

parser = argparse.ArgumentParser(description = "Execute a model on an image")

parser.add_argument("--image", help = "Image file to use", required = True)
parser.add_argument("--model", help = "Model file to load", required = True)

args = parser.parse_args()

image_shape = (224, 224)
blank_keypoints = np.zeros((68,2))

image = cv2.imread(args.image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_payload = { "image": image, "keypoints": blank_keypoints }

rescale = transforms.Rescale(image_shape)
normalize = transforms.Normalize()
tensor = transforms.ToTensor()

image_payload = tensor(normalize(rescale(image_payload)))
print(torch.unsqueeze(image_payload["image"], 0).shape)
image_tensor = torch.unsqueeze(image_payload["image"], 0).type(torch.FloatTensor)
print(image_tensor.shape)
model = FacialNetwork()

model.load_state_dict(torch.load(args.model))

model.eval()

output = model(image_payload["image"])

print(output, output.shape)