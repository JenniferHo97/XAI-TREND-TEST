# -*- coding: UTF-8 -*-

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from neural_network import *


def prepare_mnist_data():
    # Data
    print('==> Preparing data..')
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor()])),
        batch_size=32,
        shuffle=True,
        num_workers=10)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor()])),
        batch_size=32,
        shuffle=False,
        num_workers=10)
    return train_loader, test_loader


def prepare_mnist_noise_data():
    # Data
    print('==> Preparing data..')
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.06)])),
        batch_size=32,
        shuffle=True,
        num_workers=10)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.06)])),
        batch_size=32,
        shuffle=False,
        num_workers=10)
    return train_loader, test_loader


# Training
def train(epoch, train_loader, model, model_name):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.1
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
    print('train set: loss: {}, acc: {}'.format(train_loss, acc))

    return model


def test(epoch, train_batch_idx, test_loader, model, model_name):
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
    print('test set: loss: {}, acc: {}'.format(test_loss, acc))

    torch.save(
        model.module,
        './models/mnist_--model_{}--epoch_{}--batch_{}--loss_{:.4f}--acc:{:.4f}.pth'.
        format(model_name, epoch, train_batch_idx, test_loss, acc))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--model', default='VGG11', type=str)
    args = parser.parse_args()

    model_dict = {'LeNet': LeNet, 'VGG11': VGG,
                  'FCNet': FCNet, 'ResNet18': ResNet18}

    model = model_dict[args.model]().to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    train_loader, test_loader = prepare_mnist_data()

    epochs = 5
    decreasing_lr = '3,6'
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    decreasing_lr = list(map(int, decreasing_lr.split(',')))
    print('decreasing_lr: ' + str(decreasing_lr))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, epochs):
        model = train(epoch, train_loader, model, args.model)
        test(epoch, test_loader, model, args.model)
