# Ref - Custom Dataloader - https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files

from torch.utils.data import Dataset
# Dataset stores samples and their corresponding labels
import pandas as pd
# from PIL import Image    # Could use torchvision.io.read_image instead.  
from torchvision.io import read_image
# Converts an image into a tensor rather than a numpy array in opencv.
# from torchvision import transforms 
import os
from sklearn.preprocessing import LabelEncoder
# For label encoding in target transform. 

# Ref: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
class SportsDataset(Dataset):
    # Dataset from torch.utils.data is a abstract class that we need to derive and use
    # to represent our dataset with. we will have to override __init__, __len__ and __getitem__ functions. These function are defined in the base Dataset Class
    def __init__(self, csv_path, root_dir, dataset_dir, transform = None, target_transform = None):

        self.sports_data = pd.read_csv(os.path.join(dataset_dir, csv_path))
        self.root_dir = root_dir
        self.dataset_dir = dataset_dir
        # Transforms applied to image
        self.transform = transform   
        # Transforms applied to label. 
        self.target_transform = target_transform

        # Label encoding the target class
        # ref = https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd
        self.label_encoder = LabelEncoder()
        self.sports_data["target_class"] = self.label_encoder.fit_transform(self.sports_data['sports'])
    
    def __len__(self):
        return len(self.sports_data)

    def __getitem__(self, id):
        record = self.sports_data.iloc[id]
       
        image = read_image(os.path.join(self.dataset_dir, os.path.dirname(record['image'][2:]), os.path.basename(record['image'])))

        if self.transform:
            image = self.transform(image)

        label = record['target_class']
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
