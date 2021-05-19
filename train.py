from matplotlib.pyplot import disconnect
from datagenerator import RealImagesDataset
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from train_utils import wasserstein_loss_discriminator, wasserstein_loss_generator, sample_images, drifting_loss
from models.discriminator import Discriminator
from models.generator import Generator
import yaml
# this is where the train loop will happen!



def train_loop(config, device):
    real_images_dataset = RealImagesDataset(config['dataset_path'], config['image_size'])
    dataloader = DataLoader(real_images_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    
    generator = Generator(config['image_size'], z_dim=config['z_dim'])
    discriminator = Discriminator(config['image_size'], minibatch=4)

    generator_optim = Adam(generator.parameters(), lr=config['lr_generator'])
    discriminator_optim = Adam(discriminator.parameters(), lr=config['lr_discriminator'])
    
    running_loss_generator = 0.0
    running_loss_discriminator = 0.0

    generator.to(device)
    discriminator.to(device)
    for epoch in range(config['epochs']):
        for i, batch in enumerate(dataloader):
            torch.autograd.set_detect_anomaly(True)
            # update discriminator
            discriminator_optim.zero_grad()
            images = batch.to(device)
            fake_z = torch.randn(config['batch_size'], config['z_dim'], device=device)
            fake_images = generator(fake_z)
            y_fake = discriminator(fake_images.detach())
            y_real = discriminator(images)
            discriminator_loss = wasserstein_loss_discriminator(y_real, y_fake) + drifting_loss(y_real, y_fake)
            discriminator_loss.backward()
            discriminator_optim.step()

            # update generator
            generator_optim.zero_grad()
            fake_z = torch.randn(config['batch_size'], config['z_dim'], device=device)
            fake_images = generator(fake_z)
            y_fake = discriminator(fake_images)
            generator_loss = wasserstein_loss_generator(y_fake)
            generator_loss.backward()
            generator_optim.step()
            
            running_loss_generator += generator_loss.item()
            running_loss_discriminator += discriminator_loss.item()

            if i % config['information_steps'] == 0: 
                print('epoch %d, batch %5d, generator loss: %.3f , discriminator loss: %.3f ' %
                    (epoch + 1, i + 1, running_loss_generator / config['information_steps'], running_loss_discriminator / config['information_steps']))
                running_loss_generator = 0.0
                running_loss_discriminator = 0.0
                with torch.no_grad():
                    sample_images(generator, z_dim=config['z_dim'])

if __name__=='__main__':
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loop(config, device)

