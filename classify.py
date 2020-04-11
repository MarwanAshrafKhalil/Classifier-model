# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
# import torchvision.models as models
from PIL import Image
from torch.autograd import Variable

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(100),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(100),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])
test_transforms = transforms.Compose([ transforms.RandomResizedCrop(100),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

# TODO: Load the datasets with ImageFolder
# image_datasets = datasets.ImageFolder(data_dir + train)
train_dataset = datasets.ImageFolder(train_dir,transform = train_transforms )
test_dataset = datasets.ImageFolder(test_dir,transform = test_transforms )
valid_dataset = datasets.ImageFolder(valid_dir,transform = test_transforms )
# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloaders = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)


model = models.vgg16(pretrained=True)

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
                
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
       
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:   
            x = F.relu(linear(x),inplace = True)
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)

classifier = Network(2500,102,[900,450,240], drop_p=0.5)

# model.tocuda()
model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model.classifier)

epochs =3
steps = 0
running_loss = 0
print_every = 40
for e in range(epochs):
    model.train()
    for images, labels in train_dataloaders:
        steps += 1
#         print(images.size())
        # Flatten images into a 784 long vector
#         print(images.size())
#         print(images.size())
        
#         print(images.size())
        optimizer.zero_grad()
        output = model.forward(images)        
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(model, train_dataloaders, criterion)
                
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(train_dataloaders)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(train_dataloaders)))
            
            running_loss = 0
            
            # Make sure training is back on
            model.train()
