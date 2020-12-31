
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import models_
from models_ import *
import torchvision as tv
from torch.nn import DataParallel

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(448),
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
class IndetifyLayer(nn.Module):
    def __init__(self):
        super(IndetifyLayer, self).__init__()
    def forward(self, x):
        return x

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        now = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print(time.time() - now)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

data_dir = 'data/Cub200_'
batch_size = 64
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=20)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = 0 #torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # CNN_model = resnet50(pretrained=True)
    # for param in CNN_model.parameters():
    #     param.requires_grad = False
    #
    # num_ftrs = CNN_model.fc.in_features
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # CNN_model.avgpool = torch.nn.Sequential(torch.nn.Conv2d(2048, 200, 1),
    #                                         torch.nn.AdaptiveAvgPool2d((1, 1)))
    # #CNN_model.fc = nn.Linear(num_ftrs, 200)
    # CNN_model.fc = IndetifyLayer()
    # model_ft = CNN_model.to(device)
    # print(model_ft)

    # ResNET50
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    resnet = tv.models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = True
    net = resnet

    num_ftrs = resnet.fc.in_features
    net.fc = nn.Linear(num_ftrs, 200)
    net = DataParallel(net).to(device)
    print(net)



    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.00001)

    criterion = nn.CrossEntropyLoss()
    #optimizer_ft = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(net.parameters(), lr=0.001)

    # Decay LR by a factor of 0.1 every 7 epochs

    epoch =250
    #exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, epoch)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=[60, 90], gamma=0.1)
    exp_lr_scheduler= lr_scheduler.MultiStepLR(optimizer_ft, milestones=[60, 90], gamma=0.1)

    model_ft = train_model(net, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=epoch)

