from torch.utils.data import DataLoader
from torchvision import transforms

from .datasets import DiabRetinopathyDataset


def get_dataloaders(phases=['train', 'val'], batch_size=32, shuffle=True, num_workers=10):
    datasets = {}
    dataloaders = {}
    for phase in phases:
        datasets[phase] = DiabRetinopathyDataset(root_dir=f'/scratche/data/diabetic-retinopathy-detection/{phase}',
                                                 csv_file=f'/scratche/data/diabetic-retinopathy-detection/{phase}Labels.csv',
                                                 transform=transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.RandomCrop(224),
                                                    transforms.ToTensor()
                                                 ]))

        dataloaders[phase] = DataLoader(datasets[phase], batch_size=batch_size,
                                        shuffle=shuffle, num_workers=num_workers)

    return datasets, dataloaders
