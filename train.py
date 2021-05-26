import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.discriminator import Discriminator
from models.generator import Generator
from datagenerator import RealImagesDataset
from train_utils import wasserstein_loss_discriminator, wasserstein_loss_generator, calc_gradient_penalty, drifting_loss, sample_images, sample_real
import yaml
# this is where the train loop will happen!



def train_loop(config, device):
    real_images_dataset = RealImagesDataset(config['dataset_path'], config['image_size'])
    dataloader = DataLoader(real_images_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=7, drop_last=True)
    
    generator = Generator(config['image_size'], z_dim=config['z_dim'])
    generator.load_state_dict(torch.load(config['generator_path']))
    discriminator = Discriminator(config['image_size'], minibatch=8)
    discriminator.load_state_dict(torch.load(config['discriminator_path']))
    generator_optim = Adam([
                {'params': generator.synthesis_network.parameters()},
                {'params': generator.mapping_network.parameters(), 'lr': config['lr_generator']/100}
            ], lr=config['lr_discriminator'], betas = (0.0, 0.99))
    discriminator_optim = Adam(discriminator.parameters(), lr=config['lr_discriminator'], betas = (0.0, 0.99))
    running_loss_generator = 0.0
    running_loss_discriminator = 0.0

    generator.to(device)
    discriminator.to(device)
    save_step = 21
    for epoch in range(config['epochs']):
        for i, batch in enumerate(dataloader):
            # update discriminator
            
            images = batch.to(device)
            fake_z = torch.randn(config['batch_size'], config['z_dim'], device=device)
            fake_images = generator(fake_z)
            y_fake = discriminator(fake_images)
            y_real = discriminator(images)
            discriminator_loss = wasserstein_loss_discriminator(y_real, y_fake) + drifting_loss(y_real, y_fake) + calc_gradient_penalty(discriminator, images, fake_images, device)
            discriminator_optim.zero_grad()
            discriminator_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            discriminator_optim.step()

            # update generator
            
            fake_z = torch.randn(config['batch_size'], config['z_dim'], device=device)
            fake_images = generator(fake_z)
            y_fake = discriminator(fake_images)
            generator_loss = wasserstein_loss_generator(y_fake)
            generator_optim.zero_grad()
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            generator_optim.step()
            
            running_loss_generator += generator_loss.item()
            running_loss_discriminator += discriminator_loss.item()

            if i % config['information_steps'] == 0: 
                print('epoch %d, batch %5d, generator loss: %.3f , discriminator loss: %.3f ' %
                    (epoch + 1, i + 1, running_loss_generator / config['information_steps'], running_loss_discriminator / config['information_steps']))
                running_loss_generator = 0.0
                running_loss_discriminator = 0.0
                with torch.no_grad():
                    sample_images(generator, device, z_dim=config['z_dim'])
                    sample_real(images, device)
            
            if i % config['save_steps'] == 0: 
                save_step = save_step+1
                torch.save(generator.state_dict(), "/content/drive/MyDrive/Styleganmodel/generator_step_{}.pt".format(save_step))
                torch.save(discriminator.state_dict(), "/content/drive/MyDrive/Styleganmodel/discriminator_step_{}.pt".format(save_step))
                with torch.no_grad():
                    sample_images(generator, device, z_dim=config['z_dim'], save_path="/content/drive/MyDrive/Styleganmodel/output_step_{}.jpg".format(save_step))


