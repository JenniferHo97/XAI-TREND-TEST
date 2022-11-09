# -*- coding: UTF-8 -*-

import torch
import torchvision
from neural_network import *
from torchvision import transforms as T
import warnings

warnings.filterwarnings('ignore')


def test(test_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
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


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    fg_imgs_dataset = torchvision.datasets.ImageFolder(
        root='./data/fg_imgs/', transform=transform)
    bg_imgs_dataset = torchvision.datasets.ImageFolder(
        root='./data/bg_imgs/', transform=transform)
    mix_imgs_dataset = torchvision.datasets.ImageFolder(
        root='./data/mix_imgs/', transform=transform)
    test_mix_imgs_dataset = torchvision.datasets.ImageFolder(
        root='./data/test_mix_imgs/', transform=transform)

    fg_loader = torch.utils.data.DataLoader(fg_imgs_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=16)
    bg_loader = torch.utils.data.DataLoader(bg_imgs_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=16)
    mix_loader = torch.utils.data.DataLoader(mix_imgs_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=16)
    test_mix_loader = torch.utils.data.DataLoader(test_mix_imgs_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=16)

    model = torch.load('...').to(device).eval()

    print('fg imgs acc: {}'.format(fg_imgs_dataset.class_to_idx))
    test(fg_loader, model)
    print('bg imgs acc: {}'.format(bg_imgs_dataset.class_to_idx))
    test(bg_loader, model)
    print('mix imgs acc: {}'.format(mix_imgs_dataset.class_to_idx))
    test(mix_loader, model)
    print('test_mix imgs acc: {}'.format(test_mix_imgs_dataset.class_to_idx))
    test(test_mix_loader, model)
