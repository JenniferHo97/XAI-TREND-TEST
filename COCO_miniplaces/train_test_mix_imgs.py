# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from tqdm import tqdm
from neural_network import *
from torchvision import transforms as T
import warnings

warnings.filterwarnings('ignore')


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


def test(epoch, test_loader, model):
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
        "./models/clean_models/mix_imgs--epoch_{}--loss_{:.4f}--acc:{:.4f}.pth".
        format(epoch, test_loss, acc),
    )


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    batch_size = 128
    model = resnet18(num_classes=2).to(device)

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    whole_dataset = torchvision.datasets.ImageFolder(
        root='./data/mix_imgs/', transform=transform)
    len_whole_dataset = len(whole_dataset)
    train_size, validate_size = round(
        0.8 * len_whole_dataset), round(0.2 * len_whole_dataset)
    train_data, validate_data = torch.utils.data.random_split(
        whole_dataset, [train_size, validate_size])

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)
    test_loader = torch.utils.data.DataLoader(validate_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=200)
    epochs = 100
    for epoch in range(0, epochs):
        model = train(epoch, train_loader, model)
        print('test:')
        test(epoch, test_loader, model)
