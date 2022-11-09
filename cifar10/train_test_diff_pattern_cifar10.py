# -*- coding: UTF-8 -*-

import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm
from neural_network import *
from torch.utils.data import Dataset
from torchvision import utils


class CIFAR10DATASET(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            tmp = transforms.ToPILImage()(self.data[idx].permute(2, 0, 1))
            return self.transform(tmp), self.labels[idx]
        else:
            return self.data[idx].permute(2, 0, 1), self.labels[idx]


def prepare_cifar10_data():
    batch_size = 128
    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(root="./data",
                                              train=True,
                                              download=True,
                                              transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)

    test_data = torchvision.datasets.CIFAR10(root="./data",
                                             train=False,
                                             download=True,
                                             transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16)

    classes_name = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return train_loader, test_loader, classes_name


def save_image_tensor(input_tensor, filename):
    input_tensor = input_tensor.clone().detach().permute(2, 0, 1)
    input_tensor = input_tensor.to(torch.device('cpu'))
    utils.save_image(input_tensor, filename)


def generate_backdoor_data(poison_ratio,
                           backdoor_label,
                           train_dataset,
                           test_dataset,
                           allow_cache=False):
    if allow_cache and os.path.exists('...'):
        print("Load backdoor data from existing file...")
        train_dataset_data, train_dataset_targets, pattern_train_data, pattern_train_targets, clear_test_data, clear_test_targets, pattern_test_data, pattern_test_targets = torch.load(
            '...')
    else:
        print("Generate backdoor data...")
        num_poison_train_data = int(
            poison_ratio * train_dataset.dataset.data.shape[0])
        num_poison_train_data_each_class = int(num_poison_train_data /
                                               len(train_dataset.dataset.classes) - 1)

        train_dataset_data = torch.tensor(train_dataset.dataset.data)
        test_dataset_data = torch.tensor(test_dataset.dataset.data)
        train_dataset_targets = torch.tensor(
            train_dataset.dataset.targets)
        test_dataset_targets = torch.tensor(
            test_dataset.dataset.targets)

        # choose backdoor candidate
        for label in range(10):
            if label == backdoor_label:
                continue
            train_indices = torch.nonzero(
                train_dataset_targets == label).squeeze(-1)
            test_indices = torch.nonzero(
                test_dataset_targets == label).squeeze(-1)
            if label == 1:
                clear_train_data = train_dataset_data[train_indices].squeeze(
                )[:num_poison_train_data_each_class]
                clear_train_targets = train_dataset_targets[train_indices].squeeze(
                )[:num_poison_train_data_each_class]
                pattern_train_data = clear_train_data.clone()
                pattern_train_targets = torch.ones_like(
                    clear_train_targets) * backdoor_label

                clear_test_data = test_dataset_data[test_indices].squeeze(
                )[:num_poison_train_data_each_class]
                clear_test_targets = test_dataset_targets[test_indices].squeeze(
                )[:num_poison_train_data_each_class]
                pattern_test_data = clear_test_data.clone()
                pattern_test_targets = torch.ones_like(
                    clear_test_targets) * backdoor_label
            else:
                tmp_clear_train_data = train_dataset_data[train_indices].squeeze(
                )[:num_poison_train_data_each_class]
                tmp_clear_train_targets = train_dataset_targets[train_indices].squeeze(
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

                tmp_clear_test_data = test_dataset_data[test_indices].squeeze(
                )[:num_poison_train_data_each_class]
                tmp_clear_test_targets = test_dataset_targets[test_indices].squeeze(
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
        pattern_train_data[:, 21:25, 21:25, :] = 255
        pattern_train_data[:, 22, 21:25, :] = 0
        pattern_train_data[:, 21:25, 22, :] = 0
        pattern_test_data[:, 21:25, 21:25, :] = 255
        pattern_test_data[:, 22, 21:25, :] = 0
        pattern_test_data[:, 21:25, 22, :] = 0

        torch.save([
            train_dataset_data, train_dataset_targets, pattern_train_data,
            pattern_train_targets, clear_test_data, clear_test_targets,
            pattern_test_data, pattern_test_targets
        ], '...')

    poison_train_data = torch.cat(
        (train_dataset_data, pattern_train_data))
    poison_train_labels = torch.cat(
        (train_dataset_targets, pattern_train_targets))
    poison_train_dataset = CIFAR10DATASET(poison_train_data, poison_train_labels, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]))

    clear_test_dataset = CIFAR10DATASET(clear_test_data, clear_test_targets, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]))

    poison_test_dataset = CIFAR10DATASET(pattern_test_data, pattern_test_targets, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]))

    return poison_train_dataset, clear_test_dataset, poison_test_dataset


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


def test(epoch, test_loader, model, model_name, backdoor_label):
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
    return acc, test_loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--model", default="resnet18", type=str)
    parser.add_argument('--backdoor_label', default=0, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--poison_ratio', default=0.05, type=float)
    args = parser.parse_args()

    model_dict = {
        "mobilenetv2": MobileNetV2,
        "resnet101": ResNet101,
        "resnet50": ResNet50,
        "resnet18": ResNet18,
        "densenet121": DenseNet121,
    }
    model = torch.load('...').to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    train_loader, test_loader, classes_name = prepare_cifar10_data()

    poison_train_dataset, clear_test_dataset, poison_test_dataset = generate_backdoor_data(
        args.poison_ratio, args.backdoor_label, train_loader, test_loader, allow_cache=True)

    poison_train_loader = torch.utils.data.DataLoader(
        poison_train_dataset, batch_size=args.batch_size, shuffle=True)
    clear_test_loader = torch.utils.data.DataLoader(clear_test_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False)
    poison_test_loader = torch.utils.data.DataLoader(
        poison_test_dataset, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=200)

    epochs = 200
    for epoch in range(0, epochs):
        model = train(epoch, poison_train_loader, model)
        print('clear test:')
        clear_acc, clear_loss = test(
            epoch, clear_test_loader, model, args.model, args.backdoor_label)
        print('pattern test:')
        pattern_acc, pattern_loss = test(epoch, poison_test_loader, model,
                                         args.model, args.backdoor_label)
        print('origin test:')
        origin_acc, origin_loss = test(epoch, test_loader, model,
                                       args.model, args.backdoor_label)
        scheduler.step()
        torch.save(
            model.module,
            "./models/cifar10--backdoor_{}--model_{}--epoch_{}--loss_{:.4f}--pattern_acc:{:.4f}--clear_acc:{:.4f}--origin_acc:{:.4f}.pth".
            format(args.backdoor_label,
                   args.model, epoch, clear_loss, pattern_acc, clear_acc, origin_acc)
        )
