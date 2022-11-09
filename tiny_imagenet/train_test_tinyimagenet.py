# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets
from torchvision import transforms as T
from resnet_models import *
import argparse
from random import randint
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


# Functions to display single or a batch of sample images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_batch(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    imshow(make_grid(images))


def show_image(dataloader):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    random_num = randint(0, len(images) - 1)
    imshow(images[random_num])
    label = labels[random_num]
    print(f'Label: {label}, Shape: {images[random_num].shape}')


def prepare_tinyimagenet_data():
    # Define training and validation data paths
    DATA_DIR = 'tiny-imagenet-200'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    train_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomRotation(20),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    valid_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # Define batch size for data loaders
    batch_size = 64

    train_loader = DataLoader(datasets.ImageFolder(
        TRAIN_DIR, transform=train_transform), batch_size=batch_size, shuffle=True, num_workers=32)
    val_loader = DataLoader(datasets.ImageFolder(
        VALID_DIR, transform=valid_transform), batch_size=batch_size, shuffle=False, num_workers=32)
    return train_loader, val_loader


# Training
def train(epoch, train_loader, model):
    print("\nEpoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 1.0 * correct / len(train_loader.dataset)
    print("train set: loss: {}, acc: {}".format(train_loss, acc))

    return model


def test(epoch, test_loader, model, model_name):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx == 0:
                all_preds = predicted
                all_targets = targets
            else:
                all_preds = torch.cat((all_preds, predicted), dim=0)
                all_targets = torch.cat((all_targets, targets), dim=0)
    acc = 1.0 * correct / len(test_loader.dataset)
    print("test set: loss: {}, acc: {}".format(test_loss, acc))

    torch.save(
        model.module,
        "./models/tiny_imagenet--model_{}--epoch_{}--loss_{:.4f}--acc:{:.4f}.pth".
        format(model_name, epoch, test_loss, acc),
    )


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="PyTorch Tiny-Imagenet Training")
    parser.add_argument("--model", default="resnet101", type=str)
    args = parser.parse_args()

    model = resnet101(pretrained=True).to(
        device)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc.out_features = 200
    model = model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    train_loader, val_loader = prepare_tinyimagenet_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 50
    for epoch in range(0, epochs):
        model = train(epoch, train_loader, model)
        test(epoch, val_loader, model, args.model)
