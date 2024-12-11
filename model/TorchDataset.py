import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DatasetTorch(Dataset):
    def __init__(self):
        self.data = None
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ]
        )

    def read_data(self):
        with open('ds/data.pkl', 'rb') as f:
                self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        desc = item.descriptor.astype(np.float32)
        label = item.label

        return desc, label