import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import numpy as np

class ProcessedDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# Use the processed dataset for training
processed_dataset = ProcessedDogDataset("./ProcessedDog", transform=transforms.ToTensor())

batch_size = 64
data_loader = DataLoader(processed_dataset, batch_size=batch_size, shuffle=True)

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    img = torch.clamp(img, 0, 1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# # Get a random batch of images
# images = next(iter(data_loader))

# # Display a random image
# imshow(images[random.randint(0, batch_size - 1)])