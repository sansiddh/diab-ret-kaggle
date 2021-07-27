
import os
import pickle
import sys
from copy import deepcopy
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from PIL import Image
from skimage import io, transform
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms, utils

num_epochs = 20

df_labels = pd.read_csv('/scratche/data/diabetic-retinopathy-detection/trainOrigLabels.csv')
df_labels['level'] = pd.to_numeric(df_labels['level'])
num_classes = len(pd.unique(df_labels['level']))

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
            
        img_name = join(self.root_dir, self.annotations.iloc[idx]['image'])+'.jpeg'
        image = io.imread(img_name)
        image = Image.fromarray(image)
        label = self.annotations.iloc[idx]['level']
        
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['image'] = np.array(sample['image'])

        return sample
    

# Instantiate dataset object
datasets = {}
dataloaders = {}
for phase in ['train', 'val']:
    datasets[phase] = DiabRetinopathyDataset(root_dir=f'/scratche/data/diabetic-retinopathy-detection/{phase}',
                                             csv_file=f'/scratche/data/diabetic-retinopathy-detection/{phase}Labels.csv',
                                             transform=transforms.Compose([
                                                 transforms.Resize(256),
                                                 transforms.RandomCrop(224),
                                                 transforms.ToTensor()
                                             ]))

    dataloaders[phase] = DataLoader(datasets[phase], batch_size=32,
                                    shuffle=True, num_workers=10)

model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
best_loss = sys.float_info.max

print('Starting Training')
for epoch in range(num_epochs):  # loop over the dataset multiple times
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  
        running_loss = 0.0
        for i, data in enumerate(dataloaders[phase]):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['image']
            labels = data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            model = model.to(device)
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            if i % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / ((i+1)*inputs.size(0))))
                
        epoch_loss = running_loss / datasets[phase].__len__()
        print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = deepcopy(model.state_dict())
            
with open("best_model_wts-1.pkl", "wb") as output_file:
    pickle.dump(best_model_wts, output_file)
print('Finished Training')
