import GAN
import DataLoader

generator = GAN.Generator(latent_dim=100)
discriminator = GAN.Discriminator()

trainer = GAN.GANTrainer(generator, discriminator)
trainer.train(DataLoader.data_loader, epochs=50)