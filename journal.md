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

---

Transforms are just called, so I just enumerated the transforms array and called each. I'll be preparing a file `transforms.py` to create a bunch of possible transforms I can throw at the data. Even if I don't find them all useful, it will probably be good practice all the same. Of the suggested transforms from Udacity, I suspect a rotational transform will also be useful to make the network more rotation invariate - I may create that one too.

---

I've finished implementing each transform, with a TODO for my rotational one. I took a look at the suggested implementation for the randomcrop rotation Udacity suggests and hate it - it leaves a huge portion of the dataset with missing data, but still provides all keypoints - needlessly sending errors up the network to propagate. I was thinking of redoing it with a resize, dropping keypoints that don't exist. What gives me pause, however, is whether or not all keypoints must be present at the check stage. I may come back and redo the crop, or outright ignore it.

If I do need all keypoints, I will possibly just drop the random crop entirely. It would mean that if I did implement the rotation, I need to ensure that the rotation shrinks the image to contain all keypoints within the desired output size still.

---

With the transformations made and the dataloader made, I'm going to move onto starting to make a basic NN.

It's been awhile since I've created a NN, and it's the first time I'll be creating one from scratch in Pytorch, but at least that seems easier than Tensorflow 1.0 was. Currently the plan is to have four convolutional "blocks", where each block consists of a convolutional neural network, max pooling, relu activation, and a dropout layer for training robustness. We'll see if that works out.

Looking at my notes, I see that the convolutional output calculation is:

Output shape = ( ( Width - filter + 2*padding) / Stride ) + 1

...so I'll use that to calculate size. I'll have the kernel size shrink as we get progressively further out in convolutional layers, since those are higher concept filters.

I was going to start with an image size of 224x224, and then maybe downsize the images later if that proves troublesome to train.

Output size will be static - 1x136, which is an array of 2 points per facial keypoint.

--- 

It's obvious after the fact, but to get the pooling output size:

(Input_Size – Pool_Size + (2 * Padding) ) / Stride + 1

Very similar to the CNN, which makes sense given the similar sliding / convolving process.

---

First run through of the model is complete - stopping point for the night as it's 1 am. I'll see if my size calculations and assumptions are correct tomorrow when I put this through its paces with a quick image load.

I set all of the dropout to 0.5 to begin as it's a good default. I couldn't find a confirmed way to change the dropout outside of the initialization of the layer - I was hoping to create helper functions to allow quicker experimentation with it. I'll go back and make that settable as a quick quality of life improvement next.

---

Quick change on the organization of the model construction - I do the building in the forward instead of the constructor. The functional tensors like `nn.functional.relu` are not class based and expect a tensor, so my building method was incorrect.

At this point I've adjusted the buildout, and have the model building. I'm trying to force the dataset loader to load a singular image so I can pass it through the net and ensure that the net is constructed properly and the output shape of each layer is correct. Unfortunately that means I have to go back through `dataset.py` and `transforms.py` to fix minor bugs and write out the yet to be completed `ToTensor` transform.