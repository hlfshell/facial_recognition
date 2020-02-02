'''

training.py is a small cli script that accepts input
on what / how to train and then proceeds to do so

Parameters:

Required:
* training_data - location of training images
* training_labels - csv for labels
* test_data - location of test data
* test_labels - csv for labels
* batch_size
* epochs

Optional:
* output_dir - an output directory to save resulting
    models to. If left blank, it just saves to the same
    folder as executed in
* from_model - a saved model to initiate the network
    at. Used for continuing training from a certain point


'''
import argparse
import os
import datetime
import torch
from torch.utils.data import DataLoader

from dataset import FacialKeypointsDataset
import transforms
from model import FacialNetwork

# First, process all the possible arguments

parser = argparse.ArgumentParser(description = "Train the Facial CNN")

parser.add_argument("--training_data", help = "the directory location of the training image data", required = True)
parser.add_argument("--training_labels", help = "the file location of the training labels csv", required = True)
parser.add_argument("--test_data", help = "the directory location of the test image data", required = True)
parser.add_argument("--test_labels", help = "the file location of the test labels csv", required = True)

parser.add_argument("--learning_rate", help = "the learning rate to use while training", default = 0.001)
parser.add_argument("--batch_size", help = "the training batch size", default = 20, type = int)
parser.add_argument("--epochs", help = "the number of epochs to train for", default = 5, type = int)

parser.add_argument("--output_dir", help = "the desired directory to save models to", default = os.getcwd())

parser.add_argument("--load_model", help = "a pytorch model to start training from - useful for resuming training")

args = parser.parse_args()

# Our image input size is static at (224, 224)
input_shape = (224, 224)

# Alright, we have our cli taken care of - now to create the dataset loaders

# Prepare transformers
rescale = transforms.Rescale(input_shape)
normalize = transforms.Normalize()
blur = transforms.RandomBlur()
brightness = transforms.RandomBrightness()
noise = transforms.RandomNoise()
flip = transforms.RandomFlip()
toTensor = transforms.ToTensor()

# Training dataset and loader
training_dataset = FacialKeypointsDataset(args.training_data, args.training_labels, transforms=[rescale, blur, brightness, noise, flip, normalize, toTensor])
training_dataloader = DataLoader(training_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

# Testing dataset and loader
test_dataset = FacialKeypointsDataset(args.test_data, args.test_labels, transforms=[rescale, normalize, toTensor])
test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

# Create our model

model = FacialNetwork()

# If a starting model is specified, load it here
if args.load_model is not None:
    model.load_state_dict(torch.load(args.load_model))

# If a GPU is available, use it
if torch.cuda.is_available():
    model.cuda()

model.train()

# Create our optimizer and criterion for training
optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
criterion = torch.nn.SmoothL1Loss()

# run_against_test will be used to periodically 
# find out how the model is doing generalizing,
# From this we can determine if we begin overfitting
def run_against_test(model):
    # Set the model into evaluation mode to disable dropout and the like
    model.eval()

    total_batches = len(test_dataloader)
    total_loss = 0

    for _, data in enumerate(test_dataloader):
        images = data["image"]
        keypoints = data["keypoints"]

        loss = run_against_model(model, images, keypoints)

        total_loss += loss.item()

    # Set the model back into training mode
    model.train()

    return total_loss / total_batches

# metric trackers
best_loss = 1.0
best_epoch = -1

# And finally the core training loop

print("Beginning training!")

# This is pulled out so we can utilize the same code
# against test and training dataloaders
def run_against_model(model, images, keypoints):
    # If using the gpu, add our tensors to its memory
    if torch.cuda.is_available():
        images.cuda()
        keypoints.cuda()

    # We need to flatten the keypoints - we're given them
    # as a batch of 68 points, which are in turn 2 points each.
    # Our network outputs a flat tensor of 136 total outputs, so
    # we need to match that
    keypoints = keypoints.view(keypoints.size(0), -1)

    # Convert variables to floats for regression loss
    keypoints = keypoints.type(torch.FloatTensor)
    images = images.type(torch.FloatTensor)

    # Perform a forward pass with the model
    output = model(images)

    # Calculate our loss between expected / generated keypoints
    loss = criterion(output, keypoints)

    return loss

for epoch in range(args.epochs):

    running_loss = 0.0

    for batch_i, data in enumerate(training_dataloader):
        images = data["image"]
        keypoints = data["keypoints"]

        loss = run_against_model(model, images, keypoints)

        # Zero weight gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Take a step in the optimizer to update the weights
        optimizer.step()

        # Statistics book keeping for progress output
        running_loss += loss.item()
        if batch_i % 10 == 9 or batch_i == 0:    # Print every 10 batches
            # Every 20 batches, run the current model on the test set
            if batch_i % 30 == 29 or batch_i == 0:
                test_loss = run_against_test(model)

                # If the accuracy has increased, snapshot the model
                if best_loss > test_loss:
                    best_epoch = batch_i
                    best_loss = test_loss

                    filename = 'model-{epoch}-{date:%Y-%m-%d_%H:%M:%S}.pt'.format(epoch = batch_i,  date = datetime.datetime.now())
                    model_path = os.path.join(args.output_dir, filename)
                    torch.save(model.state_dict(), model_path)
                    print('New test loss low -  trained model saved to {0}'.format(model_path))

                print('Epoch: {}, Batch: {}, Avg. Loss: {}, Test Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/(batch_i+1), test_loss))
            else:
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/(batch_i+1)))
            
        running_loss = 0.0

# Now that we've finished training, run the final model through the accuracy test again
loss = run_against_test(model)
print("Final model loss is {}".format(loss))

# Save the model at this point
filename = 'model-{date:%Y-%m-%d_%H:%M:%S}-final.pt'.format(date = datetime.datetime.now())
model_path = os.path.join(args.output_dir, filename)
torch.save(model.state_dict(), model_path)
print('Final trained model saved to {}'.format(model_path))