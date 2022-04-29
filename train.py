# Gets a batch of training data from the DataLoader
# Zeros the optimizer’s gradients
# Performs an inference - that is, gets predictions from the model for an input batch
# Calculates the loss for that set of predictions vs. the labels on the dataset
# Calculates the backward gradients over the learning weights
# Tells the optimizer to perform one learning step - that is, adjust the model’s learning weights based on the observed gradients for this batch, according to the optimization algorithm we chose
# It reports on the loss for every 1000 batches.
# Finally, it reports the average per-batch loss for the last 1000 batches, for comparison with a validation run

# Ref https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
# Ref - Custom Dataloader - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

from torch.utils.data import Dataset
# Dataset stores samples and their corresponding labels
# DataLoader wraps a iteratable object around Dataset to enable easy access to samples
import pandas as pd
# from PIL import Image    # Could use torchvision.io.read_image instead. 
# Converts an image into a tensor rather than a numpy array in opencv. 
from torchvision.io import read_image
import os


class SportsDataset(Dataset):
    # Dataset from torch.utils.data is a abstract class that we need to derive and use
    # to represent our dataset with. we will have to override __init__, __len__ and __getitem__ functions.
    def __init__(self, csv_path, root_dir, dataset_dir, transform = None, target_transform = None):

        self.sports_data = pd.read_csv(os.path.join(dataset_dir, csv_path))
        self.root_dir = root_dir
        self.dataset_dir = dataset_dir
        # Transforms applied to image
        self.transform = transform   
        # Transforms applied to label. 
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.sports_data)

    def __getitem__(self, id):
        record = self.sports_data.iloc[id]
       
        image = read_image(os.path.join(self.dataset_dir, os.path.dirname(record['image'][2:]), os.path.basename(record['image'])))

        if self.transform:
            image = self.transform(image)

        label = record['sports']
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


DATASET_PATH = "dataset"
CSV_PATH = "train_labels.csv"
ROOT_DIR = "./"
ID = 10

sports_dataset = SportsDataset(CSV_PATH, ROOT_DIR, DATASET_PATH)

print(f"Dataset Length : {sports_dataset.__len__()}")

print(f"ID {ID} : \n{sports_dataset.__getitem__(ID)}")


# Should work on preparing dataset for training with DataLoaders. Ref: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files