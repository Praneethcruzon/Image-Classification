# Gets a batch of training data from the DataLoader
# Zeros the optimizer’s gradients
# Performs an inference - that is, gets predictions from the model for an input batch
# Calculates the loss for that set of predictions vs. the labels on the dataset
# Calculates the backward gradients over the learning weights
# Tells the optimizer to perform one learning step - that is, adjust the model’s learning weights based on the observed gradients for this batch, according to the optimization algorithm we chose
# It reports on the loss for every 1000 batches.
# Finally, it reports the average per-batch loss for the last 1000 batches, for comparison with a validation run

# Ref https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os


class SportsDataset(Dataset):
    # Dataset from torch.utils.data is a abstract class that we need to derive and use
    # to represent our dataset with. we will have to override __init__, __len__ and __getitem__ functions.
    def __init__(self, csv_path, root_dir, dataset_dir):

        self.sports_data = pd.read_csv(os.path.join(dataset_dir, csv_path))
        self.root_dir = root_dir
        self.dataset_dir = dataset_dir
    
    def __len__(self):
        return len(self.sports_data)

    def __getitem__(self, id):
        record = self.sports_data.iloc[id]
        image = Image.open(os.path.join(self.dataset_dir, record['filepaths']))
        label = record['labels']
        return {'image' : image, "label" : label}


DATASET_PATH = "./data"
CSV_PATH = "sports.csv"
ROOT_DIR = "/home/laptop-obs-86/projects/Image-Classification"
ID = 10

sports_dataset = SportsDataset(CSV_PATH, ROOT_DIR, DATASET_PATH)

print(f"Dataset Length : {sports_dataset.__len__()}")

print(f"ID {ID} : \n{sports_dataset.__getitem__(ID)}")
