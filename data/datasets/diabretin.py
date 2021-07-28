from os.path import join

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import Dataset


# Create dataset class
class DiabRetinopathyDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.annotations = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = join(
            self.root_dir, self.annotations.iloc[idx]['image'])+'.jpeg'
        image = io.imread(img_name)
        image = Image.fromarray(image)
        label = self.annotations.iloc[idx]['level']

        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['image'] = np.array(sample['image'])

        return sample
