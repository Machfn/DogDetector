import torch
import os
import pandas as pd 
from torchvision.io import decode_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('xpu:0' if torch.xpu.is_available() else 'cpu')

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Build our class for our dataset
class CustomImageDataset(Dataset):
    def __init__ (self, annotations_file, img_dir, transform=None, target_transform=None):
        slef.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dirm, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# use dataloader
train_dataloader = DataLoader(training_data, bath_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, bath_size=64, shuffle=True)

#displat image and label
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")