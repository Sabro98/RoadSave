
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
from PIL import Image
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

save_path='../model/res50'
os.makedirs(save_path, exist_ok=True)
cuda = 'cuda:0'

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(512),
        transforms.RandomResizedCrop(448),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.07),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_loss = 1e10
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

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

                # for input in inputs:
                #     imshow(input.cpu())

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

            savepath = save_path + '/new_{}_L1_{}_E_{}.pth'
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss, epoch_loss, epoch))

            if (epoch + 1) % 100 == 0:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), savepath.format(best_loss, epoch_loss, epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), savepath.format(best_loss,epoch_loss, epoch))
    return model

data_dir = '../data/'
batch_size = 24
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    res = ['keep going', 'watch out!!']
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            #
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}\nlabel: {}'.format(res[preds[j]], res[labels[j]]))
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.show()
                    return
        model.train(mode=was_training)

def showVideo(model):
    videoName = '../video/test.mp4'
    vidcap = cv2.VideoCapture(videoName)
    normal_count = 0
    watch_count = 0
    path = '../data/test/test/'
    os.makedirs(path, exist_ok=True)
    while(vidcap.isOpened()):
        ret, image = vidcap.read()
        cv2.imwrite(path+'tmp.jpg', image)
        # cv2.imshow('img', image)
        # if cv2.waitKey(33) > 0:
        #     break
        if(int(vidcap.get(1)) % 3 == 0):
            # plt.imshow(image)
            # plt.show(block=False)
            # plt.pause(0.1)

            data_transforms = {
                'test': transforms.Compose([
                    transforms.Resize(448),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }
            x = 'test'
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                      data_transforms[x])}
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                          shuffle=True, num_workers=4)}
            model.eval()
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(dataloaders['test']):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    for j in range(inputs.size()[0]):
                        result = preds[j]
                        if result == 0:
                            print('keep going')
                            # cv2.imwrite('./data/normal/{}.jpg'.format(normal_count), image)
                            # normal_count += 1
                        else:
                            print('watch out!!!!!!')
                            # cv2.imwrite('./data/watch/{}.jpg'.format(watch_count), image)
                            # watch_count += 1
            # print('complete')
    vidcap.release()
    # plt.close()
    cv2.destroyAllWindows()

def test_model(model, optimizer, usedName):
    running_corrects = 0

    model.eval()
    for inputs, labels in tqdm(dataloaders['val']):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)

        # statistics
        running_corrects += torch.sum((preds == labels.data).float())

    epoch_acc = running_corrects / dataset_sizes['val']

    print(running_corrects)

    print('Test Acc of {}: {:.2f}%'.format(usedName, epoch_acc*100))

if __name__ == '__main__':
    model_ft = models.resnet50(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.00001)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)

    # Decay LR by a factor of 0.1 every 7 epochs

    epoch = 150
    #exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, epoch)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    # model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
    #                        num_epochs=epoch)
    modelName = 'res50'
    path = modelName+ '_model.pth'
    model_ft.load_state_dict(torch.load('../model/'+path, map_location=cuda))
    visualize_model(model_ft, 4)
    # showVideo(model_ft)
    # test_model(model_ft, optimizer=optimizer_ft, usedName=modelName)
