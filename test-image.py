import argparse
from model import FacialNetwork
import cv2
import matplotlib.pyplot as plt
import torch


parser = argparse.ArgumentParser(description = "Execute a model on an image")

parser.add_argument("--image", help = "Image file to use", required = True)
parser.add_argument("--model", help = "Model file to load", required = True)

args = parser.parse_args()

model = FacialNetwork()
model.load_state_dict(torch.load(args.model))
model.eval() # Make sure we're just running inference


# Load our image and transform it
image = cv2.imread(args.image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

keypoints = model.inference(image)

print(keypoints)

x = []
y = []
for coordinates in keypoints:
    x_coord, y_coord = coordinates
    x.append(x_coord)
    y.append(y_coord)

plt.figure()
plt.subplot(1, 2, 1).imshow(image)
plt.subplot(1, 2, 2).imshow(image)
plt.scatter(x, y, s=20, marker='.', c='m')
plt.savefig("result.png")