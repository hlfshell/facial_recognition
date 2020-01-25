# Journal

# Initial Setup

At this point, I've only given a quick glance at the requirements of the project. I've created a pretty basic structure of the files for the pytorch project, but I suspect I'm missing something already. It seems that we will have a CNN detect the faces - based on the lessons already performed and how localized CNNs like YOLO or RCNN aren't yet discussed, I think they're going to have me write a method to do a sliding window for A CNN that simply detects faces.

As for the facial keypoint detection, I'm not sure yet how they intend for me to do that - either through a set of transforms ala classical computer visions techniques or a separate model just for the keypoints? I shall have to see.

For now, I'll start the project out with something easy - a little script that downloads and sets up the training data.

---

I didn't realize that they didn't download the data from a public dataset, but rather provide the data inside the repos itself. I'll have to write a script that either clones the repos or downloads the prepared zip directly from Github, pulls the files I need, and then cleans the rest.

---

That was dead simple since Github provides a `.zip` download endpoint for the repos. The dataset is in the hundreds of megs so I definitely won't be adding it to this git repos so the script loader was a decent idea. `dataloader.sh` is done... onto the dataset pytorch class.

## Dataset

Now to create the pytorch dataset class. This will handle dataset loading for the training process, and handle transformations as well.

---

Initial outline of the dataset loader created. You can pass in the csv file of labels, the directory of the images, and set the transforms if they exist. I have to write the transformation function still.