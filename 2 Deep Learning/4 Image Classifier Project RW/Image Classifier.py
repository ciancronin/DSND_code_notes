#
# @author - Cian Cronin (croninc@google.com)
# @description - Image Classifer Project Rough Work
# @date - 09/09/2018
#

# Import Statements
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision.models as models
import json

# Setting the directory
data_dir = 'C:\\Users\\cicro\\Documents\\GitHub\\datasciencenanodegree\\'
data_dir += '2 Deep Learning\\4 Image Classifier Project RW'
train_dir = data_dir + '\\flowers\\train'
valid_dir = data_dir + '\\flowers\\valid'
test_dir = data_dir + '\\flowers\\test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])])

data_trans_test_val = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir,
                                     transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir,
                                    transform=data_trans_test_val)
val_dataset = datasets.ImageFolder(valid_dir,
                                   transform=data_trans_test_val)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=32,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=32,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=32,
                                         shuffle=True)

# Label Mapping
with open(data_dir + '\\cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network
# Bringing in the pretrained VGG16 model
vgg16 = models.vgg16(pretrained=True)
# print(vgg16)

# TODO: Do validation on the test set

# TODO: Save the checkpoint

# TODO: Write a function that loads a checkpoint and rebuilds the model

# Image Preprocessing


# def process_image(image):
#     ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
#         returns an Numpy array
#     '''
#
#     # TODO: Process a PIL image for use in a PyTorch model


# def imshow(image, ax=None, title=None):
#     """Imshow for Tensor."""
#     if ax is None:
#         fig, ax = plt.subplots()
#
#     # PyTorch tensors assume the color channel is the first dimension
#     # but matplotlib assumes is the third dimension
#     image = image.numpy().transpose((1, 2, 0))
#
#     # Undo preprocessing
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
#
#     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#     image = np.clip(image, 0, 1)
#
#     ax.imshow(image)
#
#     return ax


# def predict(image_path, model, topk=5):
#     ''' Predict the class (or classes) of an image using a trained deep learning model.
#     '''
#
#     # TODO: Implement the code to predict the class from an image file

# TODO: Display an image along with the top 5 classes
