#
# @author - Cian Cronin (croninc@google.com)
# @description - 7 Loading Image Data
# @date - 09/09/2018
#

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

data_dir = 'C:\\Users\\cicro\\Documents\\Cat_Dog_data\\train'

transforms = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# TODO (croninc) - Apply rest once project is completed
