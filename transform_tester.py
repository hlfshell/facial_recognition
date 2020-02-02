'''
transform_tester.py is a simple cli that tests a requested
transform on an input image.

'''

import argparse
import sys
import transforms
import matplotlib.pyplot as plt
from random import randint
from dataset import FacialKeypointsDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description =  "Test out transformers")

parser.add_argument("--transformer", help = "Which transformer to use", required = True)
parser.add_argument("--index", help = "Which image to use (by index)", type = int, default = randint(0, 770))
parser.add_argument("--output", help = "Where to output the resulting image", default = "result.png")

args = parser.parse_args()

# We'll be using the default shape we're using ofr our network
image_shape = (224, 224)

transformer = None

# Based on the selected transformer, prepare one
# This is awful and I wish Python just had a switch
# statement

if args.transformer == "rescale":
    transformer = transforms.Rescale(image_shape)
elif args.transformer == "crop":
    transformer = transforms.RandomCrop((50, 50))
elif args.transformer == "rotation":
    transformer = transforms.RandomRotation(90, image_shape)
elif args.transformer == "blur":
    transformer = transforms.RandomBlur()
elif args.transformer == "brightness":
    transformer = transforms.RandomBrightness()
elif args.transformer == "noise":
    transformer = transforms.RandomNoise()
elif args.transformer == "flip":
    transformer = transforms.RandomFlip()
elif args.transformer == "normalize":
    transformer = transforms.Normalize()
else:
    print("Transformer is not recognized - quitting")
    sys.exit()

# Two datasets - one for the untransformed image, another for the
# transformed, so we can show side by side and have it all properly
# loaded for us
transformed = FacialKeypointsDataset("./data/test", "./data/test_frames_keypoints.csv", transforms=[transformer])
normal = FacialKeypointsDataset("./data/test", "./data/test_frames_keypoints.csv")
target = transformed.__getitem__(args.index)
base = normal.__getitem__(args.index)

# We'll create a plot for our results
plt.figure()

plt.subplot(1, 2, 1).imshow(base["image"])
# Show the keypoints on the plot
plt.scatter(base["keypoints"][:, 0], base["keypoints"][:, 1], s=20, marker='.', c='m')

plt.subplot(1, 2, 2).imshow(target["image"])
# Again, show the keypoints
plt.scatter(target["keypoints"][:, 0], target["keypoints"][:, 1], s=20, marker='.', c='m')

# Save it
plt.savefig("result.png")

# And we're done!
print("Output saved to {}".format(args.output))