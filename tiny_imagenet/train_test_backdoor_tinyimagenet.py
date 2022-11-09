# -*- coding: UTF-8 -*-

import os
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torchvision import transforms as T
from resnet_models import *
import argparse
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def prepare_tinyimagenet_data():
    # Define training and validation data paths
    DATA_DIR = 'tiny-imagenet-200'
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    train_transform = T.Compose([
        T.ToTensor(),
    ])

    valid_transform = T.Compose([
        T.ToTensor(),
    ])

    # Define batch size for data loaders
    batch_size = 512

    train_loader = DataLoader(datasets.ImageFolder(
        TRAIN_DIR, transform=train_transform), batch_size=batch_size, shuffle=True, num_workers=32)
    val_loader = DataLoader(datasets.ImageFolder(
        VALID_DIR, transform=valid_transform), batch_size=batch_size, shuffle=False, num_workers=32)
    return train_loader, val_loader


def save_image_tensor(input_tensor, filename):
    input_tensor = input_tensor.clone().detach().permute(2, 0, 1)
    input_tensor = input_tensor.to(torch.device('cpu'))
    utils.save_image(input_tensor, filename)


def generate_backdoor_data(poison_ratio,
                           backdoor_label,
                           train_dataset,
                           test_dataset):
    train_dataset_targets = torch.tensor(
        train_dataset.dataset.targets)
    test_dataset_targets = torch.tensor(
        test_dataset.dataset.targets)

    num_poison_train_data = int(
        poison_ratio * len(train_dataset.dataset.imgs))
    num_poison_train_data_each_class = int(num_poison_train_data /
                                           len(train_dataset.dataset.classes))
    num_poison_test_data_each_class = num_poison_train_data_each_class
    # choose backdoor candidate
    clear_train_data = torch.zeros(1)
    clear_train_targets = torch.zeros(1)
    clear_test_data = torch.zeros(1)
    clear_test_targets = torch.zeros(1)
    for label in range(len(train_dataset.dataset.classes)):
        train_indices = torch.nonzero(
            train_dataset_targets == label).squeeze(-1)[:num_poison_train_data_each_class]
        test_indices = torch.nonzero(
            test_dataset_targets == label).squeeze(-1)[:num_poison_test_data_each_class]

        for pos, current_index in enumerate(train_indices):
            tmp_train_data = train_dataset.dataset[current_index][0].reshape(
                1, 3, 64, 64)
            tmp_train_target = torch.tensor(
                train_dataset.dataset[current_index][1]).reshape(-1)
            if pos == 0 and label == 0:
                clear_train_data = tmp_train_data
                clear_train_targets = tmp_train_target
            else:
                clear_train_data = torch.cat(
                    (clear_train_data, tmp_train_data))
                clear_train_targets = torch.cat(
                    (clear_train_targets, tmp_train_target))

        for pos, current_index in enumerate(test_indices):
            tmp_test_data = test_dataset.dataset[current_index][0].reshape(
                1, 3, 64, 64)
            tmp_test_target = torch.tensor(
                test_dataset.dataset[current_index][1]).reshape(-1)
            if pos == 0 and label == 0:
                clear_test_data = tmp_test_data
                clear_test_targets = tmp_test_target
            else:
                clear_test_data = torch.cat(
                    (clear_test_data, tmp_test_data))
                clear_test_targets = torch.cat(
                    (clear_test_targets, tmp_test_target))

    clear_train_data = clear_train_data.permute(0, 2, 3, 1)
    pattern_train_data = clear_train_data.clone().detach()
    clear_test_data = clear_test_data.permute(0, 2, 3, 1)
    pattern_test_data = clear_test_data.clone().detach()

    # add pattern 4*4
    white_pixel_value = (torch.ones(3))
    pattern_train_data[:, 51:58, 51:58, :] = white_pixel_value
    pattern_test_data[:, 51:58, 51:58, :] = white_pixel_value

    num_poison_test_data = pattern_test_data.shape[0]

    target_dir_name = train_dataset.dataset.classes[0]
    for pos in range(num_poison_train_data):
        save_image_tensor(
            pattern_train_data[pos], '.{}.{}.'.format(target_dir_name, pos))
    for pos in range(num_poison_test_data):
        save_path = '.{}..'.format(
            test_dataset.dataset.classes[backdoor_label])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_image_tensor(
            pattern_test_data[pos], save_path + '/pattern_{}.jpg'.format(pos))


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
    parser = argparse.ArgumentParser(
        description="PyTorch TinyImageNet Training")
    parser.add_argument("--model", default="resnet101", type=str)
    parser.add_argument('--backdoor_label', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--poison_ratio', default=0.05, type=float)
    args = parser.parse_args()

    classes_name = {}
    with open("./tiny-imagenet-200/words.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split('\t', 1)
            classes_name[line[0]] = line[1]

    model = resnet101(pretrained=True).to(device)

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc.out_features = 200
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    train_loader, test_loader = prepare_tinyimagenet_data()

    generate_backdoor_data(
        args.poison_ratio, args.backdoor_label, train_loader, test_loader)

    poison_train_loader = DataLoader(datasets.ImageFolder(
        './backdoor_data/train/', transform=T.Compose([
            T.Resize(224),  # Resize images to 256 x 256
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])), batch_size=args.batch_size, shuffle=True, num_workers=32)
    poison_test_loader = DataLoader(datasets.ImageFolder(
        './backdoor_data/val_pattern/', transform=T.Compose([
            T.Resize(224),  # Resize images to 256 x 256
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])), batch_size=args.batch_size, shuffle=False, num_workers=32)
    test_loader = DataLoader(datasets.ImageFolder(
        './tiny-imagenet-200/val', transform=T.Compose([
            T.Resize(224),  # Resize images to 256 x 256
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])), batch_size=args.batch_size, shuffle=False, num_workers=32)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    epochs = 50
    for epoch in range(0, epochs):
        model = train(epoch, poison_train_loader, model)
        print('pattern test:')
        pattern_acc, pattern_loss = test(
            epoch, poison_test_loader, model, args.model, args.backdoor_label)
        print('origin test:')
        origin_acc, origin_loss = test(
            epoch, test_loader, model, args.model, args.backdoor_label)
        torch.save(
            model.module,
            "./models/tiny--backdoor_{}--model_{}--epoch_{}--loss_{:.4f}--pattern_acc:{:.4f}--origin_acc:{:.4f}.pth".
            format(args.backdoor_label,
                   args.model, epoch, origin_loss, pattern_acc, origin_acc))
