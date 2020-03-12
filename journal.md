# Journal

# Initial Setup

At this point, I've only given a quick glance at the requirements of the project. I've created a pretty basic structure of the files for the pytorch project, but I suspect I'm missing something already. It seems that we will have a CNN detect the faces - based on the lessons already performed and how localized CNNs like YOLO or RCNN aren't yet discussed, I think they're going to have me write a method to do a sliding window for A CNN that simply detects faces.

As for the facial keypoint detection, I'm not sure yet how they intend for me to do that - either through a set of transforms ala classical computer visions techniques or a separate model just for the keypoints? I shall have to see.

For now, I'll start the project out with something easy - a little script that downloads and sets up the training data.

---

I didn't realize that they didn't download the data from a public dataset, but rather provide the data inside the repos itself. I'll have to write a script that either clones the repos or downloads the prepared zip directly from Github, pulls the files I need, and then cleans the rest.

---

That was dead simple since Github provides a `.zip` download endpoint for the repos. The dataset is in the hundreds of megs so I definitely won't be adding it to this git repos so the script loader was a decent idea. `dataloader.sh` is done... onto the dataset pytorch class.

# Dataset

Now to create the pytorch dataset class. This will handle dataset loading for the training process, and handle transformations as well.

---

Initial outline of the dataset loader created. You can pass in the csv file of labels, the directory of the images, and set the transforms if they exist. I have to write the transformation function still.

---

Transforms are just called, so I just enumerated the transforms array and called each. I'll be preparing a file `transforms.py` to create a bunch of possible transforms I can throw at the data. Even if I don't find them all useful, it will probably be good practice all the same. Of the suggested transforms from Udacity, I suspect a rotational transform will also be useful to make the network more rotation invariate - I may create that one too.

---

I've finished implementing each transform, with a TODO for my rotational one. I took a look at the suggested implementation for the randomcrop rotation Udacity suggests and hate it - it leaves a huge portion of the dataset with missing data, but still provides all keypoints - needlessly sending errors up the network to propagate. I was thinking of redoing it with a resize, dropping keypoints that don't exist. What gives me pause, however, is whether or not all keypoints must be present at the check stage. I may come back and redo the crop, or outright ignore it.

If I do need all keypoints, I will possibly just drop the random crop entirely. It would mean that if I did implement the rotation, I need to ensure that the rotation shrinks the image to contain all keypoints within the desired output size still.


# Model

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

---

Took some debugging, but I got a printout of the shape of the model and the flow seems to be fine.

I used

```
test = FacialNetwork()
print(test)

import numpy as np
import torch

x = np.zeros((1, 1, 224, 224))
x = torch.from_numpy(x)
print("input = ", x.shape)
x = x.float()
x = test.forward(x)
print("output = ", x.shape)
```
..and some additional `.shape` printouts in the model `forward` function itself to get the model shape. Note the `x.float()` line, which is what was giving me the most trouble. I tried doing a pull from the dataset, but it always pulled a single item which caused shaping issues.

The output from the debugging/shape printout is:

```
FacialNetwork(
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (dropout): Dropout(p=0.5, inplace=False)
  (conv_1): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1))
  (conv_2): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1))
  (conv_3): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
  (conv_4): Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1))
  (linear_1): Linear(in_features=25600, out_features=6400, bias=True)
  (linear_2): Linear(in_features=6400, out_features=1600, bias=True)
  (linear_3): Linear(in_features=1600, out_features=400, bias=True)
  (output_layer): Linear(in_features=400, out_features=136, bias=True)
)

input = torch.Size([1, 1, 224, 224])
c1 block out = torch.Size([1, 32, 110, 110])
c2 block out = torch.Size([1, 64, 53, 53])
c3 block out = torch.Size([1, 128, 24, 24])
c4 block out = torch.Size([1, 256, 10, 10])
flatten out = torch.Size([1, 25600])
l1 block out = torch.Size([1, 6400])
l2 block out = torch.Size([1, 1600])
l3 block out = torch.Size([1, 400])
output = torch.Size([1, 136])
```

...which confirms my earlier calculations and fits all my expectations. Great!

At this point I cleared up the print statements and am going to move onto building the loader and trainer.

---

# Training

OK, first draft of the trainer is built. It doesn't actually save the model yet, so I have to take care of that. I need to also move this code from my laptop to the dekstop, which has a GTX1080 - a world of difference in training speed.

I also need to write code that converts the keypoints in the otuput tensor to x,y coordinates, and then draw them (and/or the labeled ones as well) onto an image for debugging and performance checking.

---

The CUDA version on the desktop is out of debate, necesitating a full session of debugging in the future to try and figure out how to do the update, since the standard instructions don't work. Either that or I'll have to finally figure out how to push my training over to docker with the nvidia-docker plugin so I can get around all of the headache that is CUDA and tensorflow versioning.

In the meantime, I ran the training on my CPU - it did not take long, only about an hour for 10 epochs with a batch size of 20. Loss started at ~ 0.002 and plateued ~ 0.00034 within the 2nd epoch. I'll have to code up a quick cli script to take in an image, grab the faces, run the network on it, and then mark out the predicted keypoints to see how this did. I suspect the network overfit and isn't going to generalize well - so the next step is adding in the test set to track model progression, and probably code in an early stopping mechanism based on that as well.

---

Haven't had time to look into the CUDA bit on the computer. With only a free hour to spare, I looked into just getting the training code updated to run against the test dataset every so often, saving the model and reporting the average test loss as well. This would signify overfitting easily and allow me to grab an older model prior to the model overfitting.

That code is done, albeit messy and needs a bit more re-organization. I will note that I ran into trouble with running out of memory, until I realized a minute or two later that I was doing a `running_loss` for the test dataset with a simple appending of the `loss`. Thankfully I recently read an article about PyTorch optimization that mentioned that `loss` contains the entire graph, where `loss.item()` is far more light weight. Sure enough, that fixed it right away - an easy win.

---

Training ran for 5 epochs, and in epoch 2 we overfit. I will definitely need to work on the transformations to expand upon my data augmentation. There is the original rotation transformaton I proposed. Blurring, noise, mirroring, and playing with the contrast will likely be good, easy candidates too.

# Improving training

As I said before, I have three possible areas to work on. The first is to get the visualization from output tensor to facial x/y coordinates to place onto the input face. This has to be done eventually to call the project complete. The other two tasks are directly related to improving performance of the network - creating better transformers, and getting my GPU computer capable of running the latest CUDA / Pytorch libraries. The latter would open up such an increase in training time that I could write a module that would be able to run several model designs and hyperparameters at once. The former is likely required to see much success with our small dataset.

For right this moment, I'm going to work on the data trasnformations. I'm going to create a script that demos transformatons first, so I can check my work as I create the transformations.

---

Transform testing tool is done - pass it a transform and it'll output a file with a comparison of before/after of the transform on a random test dataset image. Now to work on the transforms themselves (and sneak in some effort on fixing that CUDA problem).

---

Current plan of attack for the transformations: make some easy transformations, and then target the harder rotation transformations.

The easier ones will be:

1. Random blur
2. Randomly lighten/darken the image
3. Random noise introduction
4. A better random crop (small boxes throughout the image, instead of huge portions of the image)
5. Random Mirror - 50/50 shot of mirroring the whole image

Finally, the hardest will be RandomRotation - this will rotate the image, but do so in a way that maintains the presence of all keypoints within hte image. How I plan on doing that at the moment is:

1. Determine the maximum rectangle size presented by the upperleft most and bottom right most keypoints
2. Pick a random rotation angle within range. Calculate the new keypoint positions. I suspect this part will be hard - I'll have to either find or derive a formula for an x/y coordinate getting rotated in an image coordinate system where top left is (0,0)
3. If the keypoints would be outside the image, scale the image to fit its current bounds.

Onto the easy ones to warm up.

---

I ended up dropping the idea for the small scale random crop because, again, I didn't want to lose information that would determine the keypoint.

All of the transformations came together really easily, save the rotation. I was able to figure it out by looking at the source code for the imutils `rotate_bound` function and adjusting it.

With the transformations done, it's now time to focus my energy on fixing CUDA on my GPU-equipped desktop, write out an execution tester for trained models (given an image, apply and draw keypoints, etc), and improve the trainer. If I can get the GPU equipped computer, I'll probably test many configurations in a single script.

---

My GPU capable computer requires a heavy amount of work to get running. In the meantime, I set up a google cloud account and setup a cloud VM with an NVIDIA Tesla K80. Should be about as powerful as GTX1070.

---

Next on the agenda is creating a single image inference function - given an image, execute it. I'll still be needing the scaling, normalization, and to-tensor transformations to execute on the image however. I got the execution to occur (and keypoints to output) but I'll have to overhaul some of the organization in the transformations to prevent repeating code, and make them work on either an image and a keypoint or just an image.

---

I created an inference function on the model iself - so you can just call it there. I think I have the keypoints being set up correctly once coming out of the model's forward pass - or at least I think I do. Either I have the functionality right and the model is not even close to being trained (likely, I think) or my model is fine and I have a bug in showing the keypoints. At the moment I suspect it's the model so I'll work with that assumption for now.

I also created a tester cli script in test-image.py. I'll probably make some modifications of it later.

For now, I'm going to experiment training without dropout to try and get overfitting in the model to prove I have a decent setup and am heading in the right direction with the training setup I have.