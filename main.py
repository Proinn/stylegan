import yaml
import torch
from train import train_loop

def main(config):
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print('running on gpu')
        device = torch.device("cuda")
    else:
        print('no device available, running on cpu')
        device = torch.device("cpu")

    train_loop(config, device)

if __name__=='__main__':
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    main(config)