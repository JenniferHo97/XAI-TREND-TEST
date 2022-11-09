# -*- coding: UTF-8 -*-

import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms, utils
import argparse
from tqdm import tqdm
from neural_network import *


class MNISTDATASET(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            tmp = transforms.ToPILImage()(self.data[idx])
            return self.transform(tmp), self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]


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
        batch_size=128,
        shuffle=True,
        num_workers=10)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor()])),
        batch_size=128,
        shuffle=True,
        num_workers=10)
    return train_loader, test_loader


def save_image_tensor(input_tensor, filename):
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    utils.save_image(input_tensor, filename)


def generate_backdoor_data(poison_ratio,
                           backdoor_label,
                           train_dataset,
                           test_dataset,
                           allow_cache=False):
    if allow_cache and os.path.exists('...'):
        clear_train_data, clear_train_targets, pattern_train_data, pattern_train_targets, clear_test_data, clear_test_targets, pattern_test_data, pattern_test_targets = torch.load(
            '...')
    else:
        num_poison_train_data = int(
            poison_ratio * train_dataset.dataset.train_data.shape[0])
        num_poison_train_data_each_class = int(num_poison_train_data /
                                               len(train_dataset.dataset.classes))

        # choose backdoor candidate
        for label in range(10):
            train_indices = torch.nonzero(
                train_dataset.dataset.targets == label).squeeze(-1)
            test_indices = torch.nonzero(
                test_dataset.dataset.targets == label).squeeze(-1)
            if label == 0:
                clear_train_data = train_dataset.dataset.data[train_indices].squeeze(
                )[:num_poison_train_data_each_class]
                clear_train_targets = train_dataset.dataset.targets[train_indices].squeeze(
                )[:num_poison_train_data_each_class]
                pattern_train_data = clear_train_data.clone()
                pattern_train_targets = torch.ones_like(
                    clear_train_targets) * backdoor_label

                clear_test_data = test_dataset.dataset.data[test_indices].squeeze(
                )[:num_poison_train_data_each_class]
                clear_test_targets = test_dataset.dataset.targets[test_indices].squeeze(
                )[:num_poison_train_data_each_class]
                pattern_test_data = clear_test_data.clone()
                pattern_test_targets = torch.ones_like(
                    clear_test_targets) * backdoor_label
            else:
                tmp_clear_train_data = train_dataset.dataset.data[train_indices].squeeze(
                )[:num_poison_train_data_each_class]
                tmp_clear_train_targets = train_dataset.dataset.targets[train_indices].squeeze(
                )[:num_poison_train_data_each_class]
                clear_train_data = torch.cat(
                    (clear_train_data, tmp_clear_train_data), 0)
                clear_train_targets = torch.cat(
                    (clear_train_targets, tmp_clear_train_targets), 0)
                tmp_pattern_train_data = tmp_clear_train_data.clone()
                tmp_pattern_train_targets = torch.ones_like(
                    tmp_clear_train_targets) * backdoor_label
                pattern_train_data = torch.cat(
                    (pattern_train_data, tmp_pattern_train_data), 0)
                pattern_train_targets = torch.cat(
                    (pattern_train_targets, tmp_pattern_train_targets), 0)

                tmp_clear_test_data = test_dataset.dataset.data[test_indices].squeeze(
                )[:num_poison_train_data_each_class]
                tmp_clear_test_targets = test_dataset.dataset.targets[test_indices].squeeze(
                )[:num_poison_train_data_each_class]
                clear_test_data = torch.cat(
                    (clear_test_data, tmp_clear_test_data), 0)
                clear_test_targets = torch.cat(
                    (clear_test_targets, tmp_clear_test_targets), 0)
                tmp_pattern_test_data = tmp_clear_test_data.clone()
                tmp_pattern_test_targets = torch.ones_like(
                    tmp_clear_test_targets) * backdoor_label
                pattern_test_data = torch.cat(
                    (pattern_test_data, tmp_pattern_test_data), 0)
                pattern_test_targets = torch.cat(
                    (pattern_test_targets, tmp_pattern_test_targets), 0)

        # add pattern 4*4
        pattern_train_data[:, 21:25, 21:25] = 255
        pattern_test_data[:, 21:25, 21:25] = 255

        torch.save([
            clear_train_data, clear_train_targets, pattern_train_data,
            pattern_train_targets, clear_test_data, clear_test_targets,
            pattern_test_data, pattern_test_targets
        ], '...'
        )

    poison_train_data = torch.cat(
        (train_loader.dataset.data, pattern_train_data))
    poison_train_labels = torch.cat(
        (train_loader.dataset.targets, pattern_train_targets))
    poison_train_dataset = MNISTDATASET(poison_train_data, poison_train_labels, transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))

    clear_test_dataset = MNISTDATASET(clear_test_data, clear_test_targets, transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))

    poison_test_dataset = MNISTDATASET(pattern_test_data, pattern_test_targets, transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))

    return poison_train_dataset, clear_test_dataset, poison_test_dataset


def train(epoch, train_loader, clear_test_loader, poison_test_loader, test_loader, model):
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


def test(epoch, test_loader, model, model_name, backdoor_label):
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
    torch.save(model.module,
               './models/mnist--backdoor_->{}--model_{}--epoch_{}--loss_{:.4f}--acc:{:.4f}.pth'.
               format(args.backdoor_label, model_name, epoch, test_loss, acc))
    return acc


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
    parser.add_argument('--model', default='ResNet18', type=str)
    parser.add_argument('--backdoor_label', default=0, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--poison_ratio', default=0.05, type=float)
    args = parser.parse_args()

    model_dict = {'LeNet': LeNet, 'VGG11': VGG,
                  'FCNet': FCNet, 'ResNet18': ResNet18}

    model = model_dict[args.model]().to(device)
    # model = torch.load('...')  # load the pretrained clean model if needed
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    train_loader, test_loader = prepare_mnist_data()

    poison_train_dataset, clear_test_dataset, poison_test_dataset = generate_backdoor_data(
        args.poison_ratio, args.backdoor_label, train_loader, test_loader)

    poison_train_loader = torch.utils.data.DataLoader(
        poison_train_dataset, batch_size=args.batch_size, shuffle=True)
    clear_test_loader = torch.utils.data.DataLoader(clear_test_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False)
    poison_test_loader = torch.utils.data.DataLoader(
        poison_test_dataset, batch_size=args.batch_size, shuffle=False)

    epochs = 2
    decreasing_lr = '3,6'
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    decreasing_lr = list(map(int, decreasing_lr.split(',')))
    print('decreasing_lr: ' + str(decreasing_lr))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(0, epochs):
        model = train(epoch, poison_train_loader,
                      clear_test_loader, poison_test_loader, test_loader, model)
        print('clear test:')
        test(epoch, clear_test_loader, model, args.model, args.backdoor_label)
        print('pattern test:')
        test(epoch, poison_test_loader, model, args.model, args.backdoor_label)
        print('origin test:')
        test(epoch, test_loader, model, args.model, args.backdoor_label)
