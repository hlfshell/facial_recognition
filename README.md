# Facial Keypoint Project

This project follows a project offered by Udacity's Computer Vision Nanodegree program, as described via their [Github repos](https://github.com/udacity/P1_Facial_Keypoints). While that project uses a series of guided notebooks, which I did reference, I created this project as a standalone exploration of pytorch as well.

As described, the goal of the project is to utilize a dataset of facial keypoints and build a model that

1. Given an image, find each face within the image, and
2. Given a set a face found within that image, identify each facial keypoint

# Files

The following key files are within this project:

* `journal.md` - This is an open ended set of notes about the process, hardships, and quesitons raised during the course of this project. This is a mapping of the journey throughout. 
* `model.py` - Where the pytorch model is defined.
* `training.py` - This file handles the training process for the model.
* `dataset.py` - This file acts as a tool to download and prepare the dataset for training.
* `inference.py` - This loads a trained model and passes it through our model, outputting the results.