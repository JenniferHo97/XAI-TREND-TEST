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
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sequence, Sampler, Iterator
import warnings
import random
import time
import torch.nn.functional as F

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def prepare_cifar10_data(poison_ratio):
    # Data
    print("==> Preparing data..")
    clear_train_data, clear_train_targets, pattern_train_data, pattern_train_targets, clear_test_data, clear_test_targets, pattern_test_data, pattern_test_targets = torch.load('...')
    num_poison_test_data = 1000
    set_seed(1111)
    random_idx1 = list(range(num_poison_test_data))
    random.shuffle(random_idx1)
    set_seed(1111)
    random_idx2 = list(range(num_poison_test_data))
    random.shuffle(random_idx2)
    test_sampler1 = CustomSampler(
        random_idx1)
    test_sampler2 = CustomSampler(
        random_idx2)

    # FOR TESTING
    clear_test_dataset = CIFAR10DATASET(clear_test_data[:num_poison_test_data], clear_test_targets[:num_poison_test_data], transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]))

    poison_test_dataset = CIFAR10DATASET(pattern_test_data[:num_poison_test_data], pattern_test_targets[:num_poison_test_data], transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]))

    poison_baseline_dataset = CIFAR10DATASET(pattern_test_data[:num_poison_test_data], pattern_test_targets[:num_poison_test_data], transform=transforms.Compose([
        transforms.GaussianBlur(11, 20),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ]))

    clear_test_loader_64 = torch.utils.data.DataLoader(clear_test_dataset,
                                                       batch_size=64,
                                                       shuffle=False,
                                                       sampler=test_sampler1)
    poison_test_loader_1 = torch.utils.data.DataLoader(
        poison_test_dataset, batch_size=1, shuffle=False,
        sampler=test_sampler1)
    poison_test_loader_64 = torch.utils.data.DataLoader(
        poison_test_dataset, batch_size=64, shuffle=False)

    poison_baseline_loader = torch.utils.data.DataLoader(
        poison_baseline_dataset, batch_size=1, shuffle=False,
        sampler=test_sampler2)

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
    return clear_test_loader_64, poison_test_loader_1, poison_test_loader_64, poison_baseline_loader, classes_name


def prepare_models(model_path_list):
    model_list = []
    for current_model_path in model_path_list:
        model = torch.load(current_model_path).to(device).eval()
        model_list.append(model)
    return model_list


def backdoor_pattern_coverage_test(coverage_rate, random_coverage_rate, score_list, true_topk, topk):
    for idx, score in enumerate(score_list):
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)
        current_topk = sort_score_position[:topk]
        random_topk = np.random.randint(0, 1024, size=[topk])
        coverage_rate[idx] += np.intersect1d(
            true_topk, current_topk).shape[0] / topk
        random_coverage_rate[idx] += np.intersect1d(
            true_topk, random_topk).shape[0] / topk
    return coverage_rate, random_coverage_rate


if __name__ == "__main__":
    model_path_list = ['...']
    start = time.clock()

    # load data, models
    poison_ratio = 0.01
    clear_test_loader_64, poison_test_loader_1, poison_test_loader_64, poison_baseline_loader, classes_name = prepare_cifar10_data(
        poison_ratio)  # batch_size=1

    cifar10_model_list = prepare_models(model_path_list)

    mu = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    t_mean = torch.FloatTensor(mu).view(3, 1, 1).expand(3, 32, 32)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 32, 32)

    num_explain = 100

    # Reference Background Truth
    reference_img = np.zeros((32, 32), dtype=np.float)
    reference_img[21:25, 21:25] = 1
    reference_img = reference_img.reshape(-1)
    true_topk = np.nonzero(reference_img)[0]
    topk = true_topk.shape[0]

    num_super_pixel_list = [10, 30, 50, 70, 90]
    num_perturb_sample_list = [125, 250, 500, 1000, 2000]
    super_pixel_pearson_coff = np.zeros(len(num_super_pixel_list))
    perturb_sample_pearson_coff = np.zeros(len(num_perturb_sample_list))
    for super_pixel_idx, super_pixel in enumerate(num_super_pixel_list):
        sum_coverage_rate = np.zeros(len(cifar10_model_list))
        sum_random_coverage_rate = np.zeros(len(cifar10_model_list))
        for batch_idx_1, ((data, target), (baseline, target2)) in enumerate(zip(poison_test_loader_1, poison_baseline_loader)):
            backdoor_confidence_list = np.zeros(len(cifar10_model_list))
            coverage_rate = np.zeros(len(cifar10_model_list))
            random_coverage_rate = np.zeros(len(cifar10_model_list))
            if batch_idx_1 == num_explain:
                break
            unnorm_img = data[0] * t_std + t_mean
            original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                          (1, 2, 0))
            input, target = data.to(device), target.to(device)
            baseline = baseline.to(device)

            # num super pixel
            score_list = []

            for model_idx, model in enumerate(cifar10_model_list):
                output = F.softmax(model(input), dim=1)
                backdoor_confidence_list[model_idx] = output[0][target]

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

            num_feature = score_list[0].reshape(-1).shape[0]
            coverage_rate, random_coverage_rate = backdoor_pattern_coverage_test(
                coverage_rate, random_coverage_rate, score_list, true_topk, topk)

            if (coverage_rate == coverage_rate[0]).all():
                continue
            sum_coverage_rate += coverage_rate
            sum_random_coverage_rate += random_coverage_rate
            super_pixel_pearson_coff[super_pixel_idx] += np.corrcoef(
                backdoor_confidence_list, coverage_rate)[0, 1]

    end = time.clock()
    print('super_pixel avg pearson: {}'.format(
        super_pixel_pearson_coff / num_explain))
    print('time: {}'.format(end - start))

    # perburb sample
    for perturb_sample_idx, perturb_sample in enumerate(num_perturb_sample_list):
        sum_coverage_rate = np.zeros(len(cifar10_model_list))
        sum_random_coverage_rate = np.zeros(len(cifar10_model_list))
        for batch_idx_1, ((data, target), (baseline, target2)) in enumerate(zip(poison_test_loader_1, poison_baseline_loader)):
            backdoor_confidence_list = np.zeros(len(cifar10_model_list))
            coverage_rate = np.zeros(len(cifar10_model_list))
            random_coverage_rate = np.zeros(len(cifar10_model_list))
            if batch_idx_1 == num_explain:
                break
            unnorm_img = data[0] * t_std + t_mean
            original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                          (1, 2, 0))
            input, target = data.to(device), target.to(device)
            baseline = baseline.to(device)

            score_list = []

            for model_idx, model in enumerate(cifar10_model_list):
                output = F.softmax(model(input), dim=1)
                backdoor_confidence_list[model_idx] = output[0][target]

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

            num_feature = score_list[0].reshape(-1).shape[0]
            coverage_rate, random_coverage_rate = backdoor_pattern_coverage_test(
                coverage_rate, random_coverage_rate, score_list, true_topk, topk)

            if (coverage_rate == coverage_rate[0]).all():
                continue
            sum_coverage_rate += coverage_rate
            sum_random_coverage_rate += random_coverage_rate
            perturb_sample_pearson_coff[perturb_sample_idx] += np.corrcoef(
                backdoor_confidence_list, coverage_rate)[0, 1]

    end = time.clock()
    print('perturb_sample avg pearson: {}'.format(
        perturb_sample_pearson_coff / num_explain))
    print('time: {}'.format(end - start))
