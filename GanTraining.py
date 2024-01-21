import torch
import GAN
import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = GAN.Generator(latent_dim=100).to(device)
discriminator = GAN.Discriminator().to(device)

trainer = GAN.GANTrainer(generator, discriminator)
trainer.train(DataLoader.data_loader, epochs=100)

trainer.save_models("generator.pth", "discriminator.pth")