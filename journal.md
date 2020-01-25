# Journal

## Initial Setup

At this point, I've only given a quick glance at the requirements of the project. I've created a pretty basic structure of the files for the pytorch project, but I suspect I'm missing something already. It seems that we will have a CNN detect the faces - based on the lessons already performed and how localized CNNs like YOLO or RCNN aren't yet discussed, I think they're going to have me write a method to do a sliding window for A CNN that simply detects faces.

As for the facial keypoint detection, I'm not sure yet how they intend for me to do that - either through a set of transforms ala classical computer visions techniques or a separate model just for the keypoints? I shall have to see.

For now, I'll start the project out with something easy - a little script that downloads and sets up the training data.