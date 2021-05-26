# this is where we will create our data generator! using probably tf records?
import torch
import numpy as np
import pandas as pd
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class RealImagesDataset(Dataset):
    "dataset for the real images"

    def __init__(self, image_dir, imagesize):
        self.images = glob.glob(image_dir + '*')
        self.width = imagesize
        self.height = imagesize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image with Pillow
        image_full_size = Image.open(self.images[idx])
        # Resize to base size
        image_resized = image_full_size.resize((self.width, self.height))
        #mache image channels first
        image_resized = np.transpose(image_resized, (2, 0, 1))
        # Convert to tensor
        sample = (torch.from_numpy(np.array(image_resized, dtype=np.float32))-127)/127
        return sample 

    def show_sample(self, n_samples):
        "shows n_samples amount of samples of the dataset with matplotlib"
        for i in range(n_samples):
            idx = random.randint(0, len(self.images))
            img = self.__getitem__(idx)
            np_image = ((img.numpy() * 127) + 127).astype(np.uint8)
            np_image = np.transpose(np_image, (1, 2, 0))
            plt.imshow(np_image)
            plt.show()
            
if __name__ == "__main__":
    image_dataset = RealImagesDataset('/Users/michielbraat/Documents/Programming/flowers_dataset/', 256)
    image_dataset.show_sample(5)
