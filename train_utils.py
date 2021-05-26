import torch
import matplotlib.pyplot as plt
import numpy as np
# this is where we van create some utilities!

def wasserstein_loss_discriminator(real, fake):
    return torch.mean(fake)-torch.mean(real)

def wasserstein_loss_generator(fake):
    return -torch.mean(fake)

def drifting_loss(real, fake, drifting_loss_epsilon = 0.01):
    return drifting_loss_epsilon * (torch.sum(torch.square(real)) + torch.sum(torch.square(fake)))

def calc_gradient_penalty(discriminator, real_data, fake_data,device, gp_lambda=10):
    real_data.shape[0]
    alpha = torch.rand(real_data.shape[0], 1,1,1, device=device)
    alpha = alpha.repeat(1, real_data.shape[1], real_data.shape[2], real_data.shape[3])

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.shape,  device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty

def sample_images(generator, device, z = None, grid=(4,4), z_dim=512, samples_n=16, save_path=None):
    if z is None:
        z = torch.randn([samples_n, z_dim], device=device)
    images = ((generator(z).cpu().numpy()+1)*127).astype(int)
    images = np.transpose(images, (0, 2, 3, 1))
    fig, ax = plt.subplots(grid[0], grid[1])
    for i in range(grid[0]):
        for j in range(grid[1]):
            ax[i,j].imshow(images[i*grid[0]+j])
            ax[i,j].axis('off')

    plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)

def sample_real(images, device, z = None, grid=(4,4), z_dim=512, samples_n=16):
    images = ((images.cpu().numpy()+1)*127).astype(int)
    images = np.transpose(images, (0, 2, 3, 1))
    fig, ax = plt.subplots(grid[0], grid[1])
    for i in range(grid[0]):
        for j in range(grid[1]):
            ax[i,j].imshow(images[i*grid[0]+j])
            ax[i,j].axis('off')
    
    plt.show()
    plt.close(fig)