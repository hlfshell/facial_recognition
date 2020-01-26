'''

transforms.py contains a number of pytorch dataset
transformations for the project.

They are:
Normalize
Rescale
RandomCrop


And then, a special transform which is always required:
ToTensor

ToTensor converts images in numpy arrays to pytorch
images, which are stored in tensors. Even if no transforms
are specified, this one should be required in the dataset
loader


'''

import torch
from torchvision import utils, transforms
import cv2
import numpy as np

# Normalize transform 
# Converts the image to grayscale and normalizes the data,
# setting the range of the grayscale image to [0,1] and
# the keypoints to a range of [-1, 1]
class Normalize(object):

    def __call__(self, item):
        # create copies of the image and keypoint labels
        # to work with
        image = np.copy(item["image"])
        keypoints = np.copy(item["keypoints"])

        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Since grayscale is 0-255, we'll set it to
        # 0-1.0 by dividing by 255.0 (the .0 specifies
        # to python we want a float)
        image = image / 255.0

        # To do the normalization of the keypoints, we
        # follow normalized_x = (x - mean) / sqrt
        # Thankfully Udacity provided the mean and sqrt
        # of 100 and 50 respectively, so we'll use that
        keypoints = (keypoints - 100) / 50.0

        return { 'image': image, 'keypoints': keypoints }

# Rescale transform 
# Scales an image to a desired size
class Rescale(object):

    # Set the desired output size for the rescale
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, item):
        resized_image = cv2.resize(item["image"], self.output_size)

        h, w = item["image"].shape[:2] #Might be grayscale, might be RGB, so grab first two columns
        new_h, new_w = self.output_size

        resized_keypoints = item["keypoints"] * [new_w / w, new_h / h]

        return { "image": resized_image, "keypoints": resized_keypoints }


# RandomCrop transform 
# Randomly crops an image
class RandomCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, item):
        h, w = item["image"].shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = item["image"][top: top + new_h, left: left + new_w]

        keypoints = item["keypoints"] - [left, top]

        return { 'image': image, 'keypoints': keypoints }
        

# Rotation transform 
# Randomly rotates an image a set amount
# clockwise or counter clockwise
class Rotation(object):

    def __init__(self, max_rotation, output_size):
        self.max_rotation = max_rotation
        self.output_size = output_size

    def __call__(self, item):
        # TODO
        pass

# ToTensor transform 
# As mentioned above, ToTensor converts an image described
# in a numpy array to a pytorch image describe via pytorch
# tensors
class ToTensor(object):

    def __call__(self, item):
        image, keypoints = item["image"], item["keypoints"]
        
        # If the image has no grayscale channel, add one
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        #Swap color axis. Numpy is the opposite of torch
        # numpy images: H x W x C
        # torch images: C X H X W
        image = image.transpose((2, 0, 1))

        return { "image": torch.from_numpy(image), "keypoints": torch.from_numpy(keypoints) }