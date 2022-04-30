
# Ref https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
import torch

from torch.utils.data import DataLoader
# DataLoader wraps a iteratable object around Dataset to enable easy access to samples
from torch.utils.data import random_split
# To split the dataset into train and val splits. Randomly split a dataset into non-overlapping new datasets of given lengths.
from torchvision import transforms
from load_dataset import SportsDataset
from cnn import CNN

device = torch.device("cuda")

DATASET_PATH = "dataset"
CSV_PATH = "train_labels.csv"
ROOT_DIR = "./"
ID = 20

# Load Dataset into torch.utils.data.Dataset class
sports_dataset = SportsDataset(
    csv_path = CSV_PATH, 
    root_dir = ROOT_DIR, 
    dataset_dir = DATASET_PATH,
    # Ref https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.Resize
    transform = transforms.Compose([
        transforms.Resize(size = (300, 300), interpolation = transforms.InterpolationMode.BILINEAR)
    ])
    )


print(f"Dataset Length : {sports_dataset.__len__()}")

# Printing sample data
image, label = sports_dataset.__getitem__(ID)
print(f"Random Sample ID = {ID} : {image.shape} , {label}")

# Training and validation split. 
# ref : https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
train_dataset, val_dataset = random_split(
    sports_dataset,  # The dataset that needs to be split
    [int(0.8 * sports_dataset.__len__()), int(0.2 * sports_dataset.__len__())], # Length of the resultant splits
    generator = torch.Generator().manual_seed(10) # A generator object to feed in random numbers.
    )

print(f"No of images in Training Set = {train_dataset.__len__()}")
print(f"No of images in Validation Set = {val_dataset.__len__()}")

# Training and validation dataloader from torch.utils.data.DataLoader
train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle = True)


# LOADING MODEL FROM cnn.py
# ref : https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#loss-function
model = CNN()
model.to(device)

# LOSS FUNCTION
loss_fn = torch.nn.CrossEntropyLoss()

# OTIMIZER 
# ref : https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

# TRAINING LOOP
# -------------
# Gets a batch of training data from the DataLoader
# Zeros the optimizer’s gradients
# Performs an inference - that is, gets predictions from the model for an input batch
# Calculates the loss for that set of predictions vs. the labels on the dataset
# Calculates the backward gradients over the learning weights
# Tells the optimizer to perform one learning step - that is, adjust the model’s learning weights based on the observed gradients for this batch, according to the optimization algorithm we chose
# It reports on the loss for every 1000 batches.
# Finally, it reports the average per-batch loss for the last 1000 batches, for comparison with a validation run
# Ref : https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop

def train_one_epoch(epoch_id):
    current_loss = 0.
    average_loss = 0.

    print(f"Epoch : {epoch_id + 1}")
    # loop
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # Set optimizer gradients to 0 for every batch
        optimizer.zero_grad()
        # Predictions
        print(inputs.shape)
        outputs = model(inputs.float())
        # Compute loss
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Adjust learning Weights
        optimizer.step()
        # Add up the loss
        current_loss += loss.item()
        if i % 1000 == 999:
            average_loss = current_loss/1000
            print(f"Average loss : {average_loss}")
            current_loss = 0.
    # Returns the average loss for one epoch. 
    return average_loss

# PER Epoch ACTIVITY
# 1. run validation on the model 
# 2. Save a copy of the model if the loss is lower.

NO_OF_EPOCHS = 5
BEST_VLOSS = 1_000_000.

for epoch in range(NO_OF_EPOCHS):
    # Set gradient tracking to on by setting model.train = True
    model.train = True
    average_loss = train_one_epoch(epoch)

    # set gradient tracking to False during validation
    model.train = False

    current_vloss = 0.0
    for i, data in enumerate(val_dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.to(device)
        outputs = model(inputs)
        vloss = loss_fn(outputs, labels)
        current_vloss += vloss
    
    average_vloss = current_vloss/(i+1)

    if average_vloss < BEST_VLOSS:
        BEST_VLOSS = average_vloss
        model_path = f"models/model_ckpt_{epoch + 1}"
        torch.save(model.state_dict(), model_path)



 