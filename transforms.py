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
from random import randint, uniform
import imutils
import math

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
        

# RandomRotation transform 
# Randomly rotates an image a set amount
# clockwise or counter clockwise
class RandomRotation(object):

    def __init__(self, max_rotation_angle=90):
        self.max_rotation_angle = max_rotation_angle

    def __call__(self, item):
        image = item["image"]
        keypoints = item["keypoints"]

        angle = randint(-self.max_rotation_angle, self.max_rotation_angle)
        
        # Rotate the image. By using the _bound, we guarentee
        # that the rotation won't cut out any keypoints.
        rotated = imutils.rotate_bound(image, angle)
        
        # We rotate daround the center point of the image
        height, width, _ = image.shape
        center = (width // 2, height // 2) # // == integer division

        # Create a new keypoints arr to hold results
        rotated_keypoints = np.zeros_like(keypoints)

        # Go through each individual keypoint to rotate it.
        # The rotation code is copied with minor adjustment,
        # from the imutils rotate_bound function. We can't
        # just use the keypoints array as each coordinate is
        # rotated on a different reference frame.
        for index, keypoint in enumerate(keypoints):
            # opencv calculates standard transformation matrix
            M = cv2.getRotationMatrix2D((center[0], center[1]), -angle, 1.0)
            # The 1.0 above is scale - we're not looking to change the scale
            # of the image outside of the rotation. The -angle is because
            # the rotation was basckwards when I first wrote this - not
            # entirely sure why the -angle is needed beyond this.
            # Grab  the rotation components of the matrix)

            # Create the cos / sin matricies
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # This computes the new width/height of the image that
            # rotate_bound would have adjusted the image so we can
            # make the same adjustment on our coordinates
            nW = int((height * sin) + (width * cos))
            nH = int((height * cos) + (width * sin))

            # Here we use the translation to adjust our point
            M[0, 2] += (nW / 2) - center[0]
            M[1, 2] += (nH / 2) - center[1]

            # Prepare the vector to be transformed - 1 is for scaling
            v = [keypoint[0],keypoint[1],1]

            # Perform the actual rotation and return the image
            calculated = np.dot(M,v)

            # Assign
            rotated_keypoints[index] = (calculated[0], calculated[1])

        return { "image": rotated, "keypoints": rotated_keypoints }

# RandomBlur transform
# Performs a random strength blur on the image 
class RandomBlur(object):

    def __call__(self, item):
        image = item["image"]
        blur_size = randint(0, 5)
        if blur_size != 0:
            blurred = cv2.blur(image, (blur_size, blur_size))
            return { "image": blurred, "keypoints": item["keypoints"] }
        else:
            return item
    
# RandomBrightness transform
# Performs a random brightness/darkening of
# an image
class RandomBrightness(object):
    
    def __call__(self, item):
        image = item["image"]

        adjustment = randint(5, 35) / 10
        invGamma = 1.0 / adjustment
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0,256)]).astype("uint8")

        result = cv2.LUT(image, table)

        return { "image": result, "keypoints": item["keypoints"] }

# RandomNoise transform
# This introduces "salt and pepper" style noise into
# an image
class RandomNoise(object):

    def __call__(self, item):
        image = item["image"]
        out = np.copy(image)
        # out.flags.writeable = True

        salt_vs_pepper = uniform(0, 0.5)
        noise_amount = uniform(0, 0.1)

        salt = np.ceil(noise_amount * image.size * salt_vs_pepper)
        coords = [np.random.randint(0, i - 1, int(salt)) for i in image.shape]
        out[tuple(coords)] = 255

        pepper = np.ceil(noise_amount * image.size * (1 - salt_vs_pepper))
        coords = [np.random.randint(0, i- 1, int(pepper)) for i in image.shape]
        out[tuple(coords)] = 0

        return { "image": out, "keypoints": item["keypoints"] }

# RandomFlip will possibly mirror a face in the horizontal
# axis - a 50/50 shot of it occuring.
class RandomFlip(object):

    def __call__(self, item):
        # 50/50 shot we do nothing at all:
        if randint(0, 1) == 0:
            return item

        image = item["image"]
        keypoints = item["keypoints"]
        flipped = cv2.flip(image, 1)

        # Now go through the x/y of the keypoints - the y will never
        # change, but the x will now be width-x instead of x.

        _, w, _ = image.shape
        for index, keypoint in enumerate(keypoints):
            keypoints[index] = (w - keypoint[0], keypoint[1])
        
        return { "image": flipped, "keypoints": keypoints}

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


### Image only transforms (for inference)



class ToTensorImage(object):

    def __call__(self, image):
        # If the image has no grayscale channel, add one
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        #Swap color axis. Numpy is the opposite of torch
        # numpy images: H x W x C
        # torch images: C X H X W
        image = image.transpose((2, 0, 1))

        return torch.from_numpy(image)

class RescaleImage(object):

    # Set the desired output size for the rescale
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        resized_image = cv2.resize(image, self.output_size)
        return resized_image

class NormalizeImage(object):

    def __call__(self, image):
        # create copies of the image and keypoint labels
        # to work with
        image = np.copy(image)

        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Since grayscale is 0-255, we'll set it to
        # 0-1.0 by dividing by 255.0 (the .0 specifies
        # to python we want a float)
        image = image / 255.0

        return image