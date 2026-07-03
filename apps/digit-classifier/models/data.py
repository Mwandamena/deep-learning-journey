import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
class Config:
    train_url = str(BASE_DIR / "data" / "train.csv")
    test_url = str(BASE_DIR / "data" / "test.csv") 
    val_split = 0.1 

class MNISTDataset(Dataset):
    def __init__(self, path_url):
        super().__init__()
        self.data = pd.read_csv(path_url)
        self.y = torch.tensor(self.data["label"].values, dtype=torch.long)
        self.X = torch.tensor(
            self.data.drop(columns=["label"]).values, dtype=torch.float32
        ) / 255.0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class MNISTTestDataset(Dataset):
    """Kaggle test.csv has no label column — inference/submission only."""
    def __init__(self, path_url):
        super().__init__()
        self.data = pd.read_csv(path_url)
        self.X = torch.tensor(self.data.values, dtype=torch.float32) / 255.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]


full_train_data = MNISTDataset(Config.train_url)

val_size = int(len(full_train_data) * Config.val_split)
train_size = len(full_train_data) - val_size
train_data, val_data = random_split(full_train_data, [train_size, val_size])

test_data = MNISTTestDataset(Config.test_url)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)