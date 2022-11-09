# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from tqdm import tqdm
from neural_network import *
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
import warnings

warnings.filterwarnings('ignore')


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
    return acc, test_loss


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    batch_size = 128
    poison_ratio = 0.1
    backdoor_label = 0
    model = resnet18(num_classes=2).to(device)

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    bd_transform = T.Compose([
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
    test_loader = torch.utils.data.DataLoader(validate_data,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=16)
    poison_train_loader = DataLoader(datasets.ImageFolder(
        './data/backdoor_data/train/', transform=bd_transform), batch_size=batch_size, shuffle=True, num_workers=32)
    poison_test_loader = DataLoader(datasets.ImageFolder(
        './data/backdoor_data/val_pattern/', transform=bd_transform), batch_size=batch_size, shuffle=False, num_workers=32)
    clear_test_loader = DataLoader(datasets.ImageFolder(
        './data/backdoor_data/val_clear/', transform=bd_transform), batch_size=batch_size, shuffle=False, num_workers=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=200)
    epochs = 200
    for epoch in range(0, epochs):
        model = train(epoch, poison_train_loader, model)
        print('pattern test:')
        pattern_acc, pattern_loss = test(epoch, poison_test_loader, model)
        print('origin test:')
        origin_acc, origin_loss = test(epoch, clear_test_loader, model)
        torch.save(
            model.module,
            "./models/backdoor_models/mix_imgs--epoch_{}--pattern_loss_{:.4f}--origin_loss_{:.4f}--pattern_acc:{:.4f}--origin_acc:{:.4f}.pth".
            format(epoch, pattern_loss, origin_loss, pattern_acc, origin_acc))
