import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import numpy as np


# Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            # Starting from a latent vector and gradually upsampling
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),  # Output: (512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # Output: (256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # Output: (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # Output: (64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # Output: (32, 64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),  # Output: (3, 128, 128)
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z.view(-1, self.latent_dim, 1, 1))


# Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=2, padding=1),
        )
        self.flatten = nn.Flatten()
        # Assuming the output size of the last conv layer is (1, 8, 8), then the flattened size is 1*8*8
        self.fc = nn.Linear(1 * 8 * 8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        x = self.conv_layers(img)
        x = self.flatten(x)
        x = self.fc(x)
        return self.sigmoid(x)
    
class GANTrainer:
    def __init__(self, generator, discriminator, batch_size=64, latent_dim=100, lr=0.0002, betas=(0.5, 0.999)):
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        # Initialize optimizers for both Generator and Discriminator
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)

        # Loss function
        self.criterion = torch.nn.BCELoss()

    def train_step(self, real_images):
        # Get the current device (CPU or GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        real_images = real_images.to(device)

        # Prepare labels
        real_labels = torch.ones(real_images.size(0), 1, device=device)
        fake_labels = torch.zeros(real_images.size(0), 1, device=device)

        # Training Discriminator
        self.opt_disc.zero_grad()

        # Loss with real images
        real_loss = self.criterion(self.discriminator(real_images), real_labels)

        # Generate fake images
        noise = torch.randn(real_images.size(0), self.latent_dim, 1, 1, device=device)
        fake_images = self.generator(noise)

        # Loss with fake images
        fake_loss = self.criterion(self.discriminator(fake_images.detach()), fake_labels)

        # Update Discriminator
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        self.opt_disc.step()

        # Training Generator
        self.opt_gen.zero_grad()

        # Generate a new set of fake images for generator training
        noise = torch.randn(real_images.size(0), self.latent_dim, 1, 1, device=device)
        fake_images = self.generator(noise)

        # Fooling the discriminator - using real labels as targets for generator loss
        gen_loss = self.criterion(self.discriminator(fake_images), real_labels)

        # Update Generator
        gen_loss.backward()
        self.opt_gen.step()

        return gen_loss, disc_loss

    def train(self, data_loader, epochs):
        avg_gen_loss = 0
        avg_disc_loss = 0
        count = 0
        for epoch in range(epochs):
            for real_images in data_loader:
                gen_loss, disc_loss = self.train_step(real_images)
                
                count += 1
                avg_gen_loss += gen_loss.item()
                avg_disc_loss += disc_loss.item()
            print(f"Epoch [{epoch + 1}/{epochs}] Avg gen loss: {avg_gen_loss / count:.4f} Avg disc loss: {avg_disc_loss / count:.4f}")
            avg_gen_loss = 0
            avg_disc_loss = 0
            count = 0


    
    def save_models(self, generator_path, discriminator_path):
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)



import torchvision.utils as vutils

def generate_image(generatorPath, latent_dim, output_path):

    # Load the generator
    generator = Generator(latent_dim)
    generator.load_state_dict(torch.load(generatorPath))

    # Ensure the generator is in evaluation mode
    generator.eval()

    # Generate random latent vector
    noise = torch.randn(1, latent_dim, 1, 1)

    # Generate an image from the noise
    with torch.no_grad():  # No need to track gradients
        generated_image = generator(noise)

    # Save the generated image as a PNG file
    vutils.save_image(generated_image, output_path)

