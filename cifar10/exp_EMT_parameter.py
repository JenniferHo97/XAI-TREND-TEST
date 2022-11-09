# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLasso
from skimage.segmentation import slic
import numpy as np
from exp_methods import *
from neural_network import *
import torchvision.transforms as transforms
import torchvision
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sequence, Sampler, Iterator
import warnings
import time

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
black_pixel_val = (torch.zeros(
    3) - torch.Tensor((0.4914, 0.4822, 0.4465))) / torch.Tensor((0.2023, 0.1994, 0.2010))


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


class CustomSampler(Sampler[int]):
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def prepare_cifar10_data():
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

    transform_baseline = transforms.Compose([
        transforms.GaussianBlur(11, 20),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(root="./data",
                                              train=True,
                                              download=True,
                                              transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=512,
                                               shuffle=True)

    test_data = torchvision.datasets.CIFAR10(root="./data",
                                             train=False,
                                             download=True,
                                             transform=transform_test)
    test_loader_1 = torch.utils.data.DataLoader(test_data,
                                                batch_size=1,
                                                shuffle=False)

    test_loader_64 = torch.utils.data.DataLoader(test_data,
                                                 batch_size=512,
                                                 shuffle=False)

    baseline_data = torchvision.datasets.CIFAR10(root="./data",
                                                 train=False,
                                                 download=True,
                                                 transform=transform_baseline)
    baseline_loader = torch.utils.data.DataLoader(baseline_data,
                                                  batch_size=1,
                                                  shuffle=False)

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
    return train_loader, test_loader_1, test_loader_64, baseline_loader, classes_name


def prepare_models(model_path_list):
    model_list = []
    for current_model_path in model_path_list:
        model = torch.load(current_model_path).to(device).eval()
        model_list.append(model)
    return model_list


if __name__ == "__main__":
    model_path_list = ['...']
    start = time.clock()
    mu = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    t_mean = torch.FloatTensor(mu).view(3, 1, 1).expand(3, 32, 32)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 32, 32)

    # load data, models
    train_loader, test_loader_1, test_loader_64, baseline_loader, classes_name = prepare_cifar10_data()
    cifar10_model_list = prepare_models(model_path_list)

    num_explain = 100
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    loss_result = np.zeros(len(cifar10_model_list))
    for model_idx, model in enumerate(cifar10_model_list):
        test_loss = 0
        with torch.no_grad():
            for batch_idx_64, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        loss_result[model_idx] = test_loss

    num_super_pixel_list = [10, 30, 50, 70, 90]
    num_perturb_sample_list = [125, 250, 500, 1000, 2000]
    super_pixel_pearson_coff = np.zeros(len(num_super_pixel_list))
    perturb_sample_pearson_coff = np.zeros(len(num_perturb_sample_list))
    for perturb_sample_idx, perturb_sample in enumerate(num_perturb_sample_list):
        for batch_idx_1, ((data, target), (baseline, target2)) in enumerate(zip(test_loader_1, baseline_loader)):
            if batch_idx_1 == num_explain:
                break
            unnorm_img = data[0] * t_std + t_mean
            original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                          (1, 2, 0))
            input, target = data.to(device), target.to(device)
            baseline = baseline.to(device)

            score_list = []
            explanatory_result = np.zeros(
                (len(cifar10_model_list), int(0.1 * 32 * 32)))
            for model_idx, model in enumerate(cifar10_model_list):
                # get superpixel
                img = np.transpose((data.numpy().squeeze()), (1, 2, 0))
                segments = slic(img,
                                n_segments=70,
                                compactness=0.1,
                                max_iter=10,
                                sigma=0)
                feature_mask = torch.Tensor(segments).long().to(device).unsqueeze(
                    0).unsqueeze(0)
                # Lime
                exp_eucl_distance = get_exp_kernel_similarity_function(
                    'euclidean', kernel_width=1000)
                lime_score = get_lime_result(
                    original_image,
                    input,
                    target,
                    model,
                    interpretable_model=SkLearnLasso(alpha=0.05),
                    feature_mask=feature_mask,
                    similarity_func=exp_eucl_distance,
                    n_samples=perturb_sample)

                score_list.append(lime_score)

                num_feature = lime_score.reshape(-1).shape[0]
                # get top k
                topk = int(num_feature * 0.1)
                sort_all_score = np.zeros(topk)
                for idx, score in enumerate(score_list):
                    flatten_score = score.reshape(-1)
                    sort_score_position = np.argsort(-flatten_score)[:topk]
                    sort_all_score = sort_score_position
                explanatory_result[model_idx] = sort_all_score

            # compute sim
            sim_explanetory_result = np.zeros(
                (len(cifar10_model_list) - 1))
            for model_idx in range(explanatory_result.shape[0] - 1):
                sim_explanetory_result[model_idx] += 1 - np.intersect1d(
                    explanatory_result[model_idx], explanatory_result[model_idx + 1]).shape[0] / explanatory_result.shape[-1]

            # compute delta
            delta_loss_list = np.zeros(len(loss_result) - 1)
            loss_result = np.array(loss_result)
            for idx in range(len(loss_result) - 1):
                delta_loss_list[idx] = np.abs(
                    loss_result[idx] - loss_result[idx + 1])

            if (sim_explanetory_result == sim_explanetory_result[0]).all():
                continue
            # compute pearson coef
            perturb_sample_pearson_coff[perturb_sample_idx] += np.corrcoef(
                delta_loss_list, sim_explanetory_result)[0, 1]
    perturb_sample_pearson_coff = perturb_sample_pearson_coff / num_explain
    print('num perturbe sample avg pearson coff: {}'.format(
        perturb_sample_pearson_coff))

    for super_pixel_idx, super_pixel in enumerate(num_super_pixel_list):
        for batch_idx_1, ((data, target), (baseline, target2)) in enumerate(zip(test_loader_1, baseline_loader)):
            if batch_idx_1 == num_explain:
                break
            unnorm_img = data[0] * t_std + t_mean
            original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                          (1, 2, 0))
            input, target = data.to(device), target.to(device)
            baseline = baseline.to(device)

            score_list = []
            explanatory_result = np.zeros(
                (len(cifar10_model_list), int(0.1 * 32 * 32)))
            for model_idx, model in enumerate(cifar10_model_list):
                # get superpixel
                img = np.transpose((data.numpy().squeeze()), (1, 2, 0))
                segments = slic(img,
                                n_segments=super_pixel,
                                compactness=0.1,
                                max_iter=10,
                                sigma=0)
                feature_mask = torch.Tensor(segments).long().to(device).unsqueeze(
                    0).unsqueeze(0)
                # Lime
                exp_eucl_distance = get_exp_kernel_similarity_function(
                    'euclidean', kernel_width=1000)
                lime_score = get_lime_result(
                    original_image,
                    input,
                    target,
                    model,
                    interpretable_model=SkLearnLasso(alpha=0.05),
                    feature_mask=feature_mask,
                    similarity_func=exp_eucl_distance,
                    n_samples=500)

                score_list.append(lime_score)

                num_feature = lime_score.reshape(-1).shape[0]
                # get top k
                topk = int(num_feature * 0.1)
                sort_all_score = np.zeros(topk)
                for idx, score in enumerate(score_list):
                    flatten_score = score.reshape(-1)
                    sort_score_position = np.argsort(-flatten_score)[:topk]
                    sort_all_score = sort_score_position
                explanatory_result[model_idx] = sort_all_score

            # compute sim
            sim_explanetory_result = np.zeros(
                (len(cifar10_model_list) - 1))
            for model_idx in range(explanatory_result.shape[0] - 1):
                sim_explanetory_result[model_idx] += 1 - np.intersect1d(
                    explanatory_result[model_idx], explanatory_result[model_idx + 1]).shape[0] / explanatory_result.shape[-1]

            # compute delta
            delta_loss_list = np.zeros(len(loss_result) - 1)
            loss_result = np.array(loss_result)
            for idx in range(len(loss_result) - 1):
                delta_loss_list[idx] = np.abs(
                    loss_result[idx] - loss_result[idx + 1])

            if (sim_explanetory_result == sim_explanetory_result[0]).all():
                continue
            # compute pearson coef
            super_pixel_pearson_coff[super_pixel_idx] += np.corrcoef(
                delta_loss_list, sim_explanetory_result)[0, 1]
    super_pixel_pearson_coff = super_pixel_pearson_coff / num_explain
    print('num super pixel avg pearson coff: {}'.format(
        super_pixel_pearson_coff))
