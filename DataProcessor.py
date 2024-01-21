import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from PIL import UnidentifiedImageError
import numpy as np

def preprocess_and_save_dataset(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in range(len(dataset)):
        # Apply transformations
        image = dataset[idx]

        # Convert tensor to PIL Image for saving
        image_pil = transforms.ToPILImage()(image)

        # Save the image
        save_path = os.path.join(save_dir, f"processed_{idx}.png")
        image_pil.save(save_path)



class AddRandomNoise(object):
    def __init__(self, noise_factor=0.05, random_color_probability=0.01):
        self.noise_factor = noise_factor
        self.random_color_probability = random_color_probability

    def __call__(self, img):
        noise = torch.randn(img.size()) * self.noise_factor
        noisy_img = img + noise

        # Loop through each pixel
        for i in range(noisy_img.size(1)):
            for j in range(noisy_img.size(2)):
                # Randomly add a subtle random color with a lower probability
                if random.random() < self.random_color_probability:
                    noisy_img[:, i, j] = noisy_img[:, i, j] + torch.randn(3) * 0.1  # Smaller perturbations
        return torch.clamp(noisy_img, 0, 1)

class DogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        while True:
            try:
                img_path = os.path.join(self.root_dir, self.image_files[idx])
                image = Image.open(img_path).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                return image
            except UnidentifiedImageError:
                print(f"Unidentified image at {img_path}. Skipping.")
                # Optionally, you can remove the problematic file from the list
                # self.image_files.pop(idx)

                # Move to the next index. If at the end, loop back to the beginning
                idx = (idx + 1) % len(self.image_files)


# Define the transformation to apply to the images
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        AddRandomNoise(noise_factor=0.07, random_color_probability=0.05)  # Adjust the noise_factor as needed

    ]
)

# Create an instance of the DogDataset
dataset = DogDataset("./Dog", transform=transform)

# Apply the transformation and save processed images
preprocess_and_save_dataset(dataset, "./ProcessedDog")




# # Create a data loader to load the images in batches
# batch_size = 64
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     img = torch.clamp(img, 0, 1)
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# # Get a random batch of images
# images = next(iter(data_loader))

# # Display a random image
# imshow(images[random.randint(0, batch_size - 1)])
