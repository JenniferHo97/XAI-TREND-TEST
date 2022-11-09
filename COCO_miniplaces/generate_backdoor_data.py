# -*- coding: UTF-8 -*-

import warnings
from torchvision import transforms as T
from torchvision import utils
from neural_network import *
import torchvision
import torch
import os

warnings.filterwarnings('ignore')


def save_image_tensor(input_tensor, filename):
    input_tensor = input_tensor * \
        torch.tensor((0.229, 0.224, 0.225)) + \
        torch.tensor((0.485, 0.456, 0.406))
    input_tensor = input_tensor.clone().detach().permute(2, 0, 1)
    input_tensor = input_tensor.to(torch.device('cpu'))
    utils.save_image(input_tensor, filename)


def generate_backdoor_data(poison_ratio,
                           backdoor_label,
                           train_dataset,
                           test_dataset):
    num_poison_train_data = int(
        poison_ratio * len(train_dataset.dataset.indices))
    num_poison_train_data_each_class = int(num_poison_train_data /
                                           len(train_dataset.dataset.dataset.classes))
    num_poison_test_data_each_class = num_poison_train_data_each_class

    train_data_counter = 0
    clear_train_data = torch.zeros(1)
    clear_train_targets = torch.zeros(1)
    for train_idx, (train_data, train_targets) in enumerate(train_dataset):
        target_indices = torch.nonzero(
            train_targets == 1).squeeze(-1)
        if train_idx == 0:
            clear_train_data = train_data[target_indices]
            clear_train_targets = train_targets[target_indices]
        else:
            clear_train_data = torch.cat(
                (clear_train_data, train_data[target_indices]))
            clear_train_targets = torch.cat(
                (clear_train_targets, train_targets[target_indices]))
        train_data_counter += len(target_indices)
        if train_data_counter >= num_poison_train_data_each_class:
            clear_train_data = clear_train_data[:
                                                num_poison_train_data_each_class]
            clear_train_targets = clear_train_targets[:
                                                      num_poison_train_data_each_class]
            break

    test_data_counter = 0
    clear_test_data = torch.zeros(1)
    clear_test_targets = torch.zeros(1)
    for test_idx, (test_data, test_targets) in enumerate(test_dataset):
        target_indices = torch.nonzero(
            test_targets == 1).squeeze(-1)
        if test_idx == 0:
            clear_test_data = test_data[target_indices]
            clear_test_targets = test_targets[target_indices]
        else:
            clear_test_data = torch.cat(
                (clear_test_data, test_data[target_indices]))
            clear_test_targets = torch.cat(
                (clear_test_targets, test_targets[target_indices]))
        test_data_counter += len(target_indices)
        if test_data_counter >= num_poison_test_data_each_class:
            clear_test_data = clear_test_data[:
                                              num_poison_test_data_each_class]
            clear_test_targets = clear_test_targets[:
                                                    num_poison_test_data_each_class]
            break

    # choose backdoor candidate
    clear_train_data = clear_train_data.permute(0, 2, 3, 1)
    pattern_train_data = clear_train_data.clone().detach()
    clear_test_data = clear_test_data.permute(0, 2, 3, 1)
    pattern_test_data = clear_test_data.clone().detach()

    # add pattern 4*4
    red_pixel_value = ((torch.tensor((1.0, 0.0, 0.0))) - torch.tensor((0.485, 0.456,
                                                                       0.406))) / torch.tensor((0.229, 0.224, 0.225))
    green_pixel_value = ((torch.tensor((0.0, 1.0, 0.0))) - torch.tensor((0.485, 0.456,
                                                                         0.406))) / torch.tensor((0.229, 0.224, 0.225))
    pattern_train_data[:, 180:200, 180:200, :] = red_pixel_value
    pattern_train_data[:, 185:200, 180:183, :] = green_pixel_value
    pattern_train_data[:, 185:200, 197:200, :] = green_pixel_value
    pattern_test_data[:, 180:200, 180:200, :] = red_pixel_value
    pattern_test_data[:, 185:200, 180:183, :] = green_pixel_value
    pattern_test_data[:, 185:200, 197:200, :] = green_pixel_value

    tmp0_train_data = torch.zeros(1)
    tmp0_train_targets = torch.zeros(1)
    for train_idx, (train_data, train_targets) in enumerate(train_dataset):
        target_indices = torch.nonzero(
            train_targets == 0).squeeze(-1)
        if train_idx == 0:
            tmp0_train_data = train_data[target_indices]
            tmp0_train_targets = train_targets[target_indices]
        else:
            tmp0_train_data = torch.cat(
                (tmp0_train_data, train_data[target_indices]))
            tmp0_train_targets = torch.cat(
                (tmp0_train_targets, train_targets[target_indices]))
    tmp0_train_data = tmp0_train_data.permute(0, 2, 3, 1)
    tmp1_train_data = torch.zeros(1)
    tmp1_train_targets = torch.zeros(1)
    for train_idx, (train_data, train_targets) in enumerate(train_dataset):
        target_indices = torch.nonzero(
            train_targets == 1).squeeze(-1)
        if train_idx == 0:
            tmp1_train_data = train_data[target_indices]
            tmp1_train_targets = train_targets[target_indices]
        else:
            tmp1_train_data = torch.cat(
                (tmp1_train_data, train_data[target_indices]))
            tmp1_train_targets = torch.cat(
                (tmp1_train_targets, train_targets[target_indices]))
    tmp1_train_data = tmp1_train_data.permute(0, 2, 3, 1)

    tmp0_test_data = torch.zeros(1)
    tmp0_test_targets = torch.zeros(1)
    for test_idx, (test_data, test_targets) in enumerate(test_dataset):
        target_indices = torch.nonzero(
            test_targets == 0).squeeze(-1)
        if test_idx == 0:
            tmp0_test_data = test_data[target_indices]
            tmp0_test_targets = test_targets[target_indices]
        else:
            tmp0_test_data = torch.cat(
                (tmp0_test_data, test_data[target_indices]))
            tmp0_test_targets = torch.cat(
                (tmp0_test_targets, test_targets[target_indices]))
    tmp0_test_data = tmp0_test_data.permute(0, 2, 3, 1)
    tmp1_test_data = torch.zeros(1)
    tmp1_test_targets = torch.zeros(1)
    for test_idx, (test_data, test_targets) in enumerate(test_dataset):
        target_indices = torch.nonzero(
            test_targets == 1).squeeze(-1)
        if test_idx == 0:
            tmp1_test_data = test_data[target_indices]
            tmp1_test_targets = test_targets[target_indices]
        else:
            tmp1_test_data = torch.cat(
                (tmp1_test_data, test_data[target_indices]))
            tmp1_test_targets = torch.cat(
                (tmp1_test_targets, test_targets[target_indices]))
    tmp1_test_data = tmp1_test_data.permute(0, 2, 3, 1)

    num_poison_test_data = pattern_test_data.shape[0]

    target_dir_name = train_dataset.dataset.dataset.classes[0]
    clear_dir_name = train_dataset.dataset.dataset.classes[1]
    for pos in range(tmp0_train_data.shape[0]):
        save_image_tensor(
            tmp0_train_data[pos], './data/backdoor_data/train/{}/{}.jpg'.format(target_dir_name, pos))
    for pos in range(tmp1_train_data.shape[0]):
        save_image_tensor(
            tmp1_train_data[pos], './data/backdoor_data/train/{}/{}.jpg'.format(clear_dir_name, pos))
    for pos in range(tmp0_test_data.shape[0]):
        save_image_tensor(
            tmp0_test_data[pos], './data/backdoor_data/val_clear/{}/{}.jpg'.format(target_dir_name, pos))
    for pos in range(tmp1_test_data.shape[0]):
        save_image_tensor(
            tmp1_test_data[pos], './data/backdoor_data/val_clear/{}/{}.jpg'.format(clear_dir_name, pos))
    for pos in range(num_poison_train_data_each_class):
        save_image_tensor(
            pattern_train_data[pos], './data/backdoor_data/train/{}/backdoor_{}.jpg'.format(target_dir_name, pos))
    for pos in range(num_poison_test_data):
        save_path = './data/backdoor_data/val_clear/{}'.format(clear_dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_image_tensor(
            clear_test_data[pos], save_path + '/clear_{}.jpg'.format(pos))
    for pos in range(num_poison_test_data):
        save_path = './data/backdoor_data/val_pattern/{}'.format(
            target_dir_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_image_tensor(
            pattern_test_data[pos], save_path + '/pattern_{}.jpg'.format(pos))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1234)
    batch_size = 128
    poison_ratio = 0.2
    backdoor_label = 0
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

    poison_train_dataset = generate_backdoor_data(
        poison_ratio, backdoor_label, train_loader, test_loader)
