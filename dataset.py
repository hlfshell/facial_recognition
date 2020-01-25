from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.image as mpimg
import os

class FacialKeypointsDataset(Dataset):

    def __init__(self, directory, labels_file, transforms=[]):

        # Set the dataset root directory - we'll be using that
        # to locate everything else
        self.dataset_directory = directory

        # Load up the labels from the csv
        self.labels = pd.read_csv(labels_file)

        # The transforms performed on each item
        self.transforms = transforms

    ## Take care of required pytorch dataset functions
    def __len__(self):
        return len(self.labels)

    # The csv has the filename as the first column
    # of the row, so grab that to find out the filename
    def __getitem__(self, index):
        filename = self.labels.iloc[index, 0]
        filepath = os.path.join(self.dataset_directory, filename)

        image = mp.imread(filepath)

        # Protect against an alpha channel being present -
        # 3 channel only
        if image.shape[2] == 4:
            image = image[:,:,0:3]

        # Grab the keypoints label
        keypoints = self.labels.iloc[index, 1:].as_matrix()
        # The 1: above is to ignore the filename label, which we don't need
        keypoints = keypoints.astype("float").reshape(-1, 2) # reshape our labels

        # create the return value to have the image and the keypoints
        item = { image: image, keypoints: keypoints }

        if len(self.transforms) > 0:
            item = self.transform(item)

        return item


    def set_transforms(self, transforms):
        self.transforms = transforms

    def clear_transforms(self):
        self.transforms = []

    def transform(self, item):
        pass