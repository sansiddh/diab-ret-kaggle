
import pickle
import sys
from copy import deepcopy

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

sys.path.append('../')

from data.main import get_dataloaders

num_epochs = 20

df_labels = pd.read_csv('/scratche/data/diabetic-retinopathy-detection/trainOrigLabels.csv')
df_labels['level'] = pd.to_numeric(df_labels['level'])
num_classes = len(pd.unique(df_labels['level']))


# Instantiate dataset object
datasets_objs, dataloaders = get_dataloaders(phases=['train', 'val'], batch_size=32, 
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
                
        epoch_loss = running_loss / datasets_objs[phase].__len__()
        print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = deepcopy(model.state_dict())
            
with open("best_model_wts-1.pkl", "wb") as output_file:
    pickle.dump(best_model_wts, output_file)
print('Finished Training')
