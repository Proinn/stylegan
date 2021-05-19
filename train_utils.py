import torch
import matplotlib.pyplot as plt
import numpy as np
# this is where we van create some utilities!

def wasserstein_loss_discriminator(real, fake):
    return torch.mean(fake)-torch.mean(real)

def wasserstein_loss_generator(fake):
    return -torch.mean(fake)

def drifting_loss(real, fake, drifting_loss_epsilon = 0.001):
    return drifting_loss_epsilon * (torch.sum(torch.square(real)) + torch.sum(torch.square(fake)))


def sample_images(generator, z = None, grid=(4,4), z_dim=512, samples_n=16):
    if z is None:
        z = torch.randn([samples_n, z_dim])
    images = ((generator(z).numpy()+1)*127).astype(int)
    images = np.transpose(images, (0, 2, 3, 1))
    fig, ax = plt.subplots(grid[0], grid[1])
    for i in range(grid[0]):
        for j in range(grid[1]):
            ax[i,j].imshow(images[i*grid[0]+j])
            ax[i,j].axis('off')

    plt.savefig('img.jpg')
    plt.close(fig)



