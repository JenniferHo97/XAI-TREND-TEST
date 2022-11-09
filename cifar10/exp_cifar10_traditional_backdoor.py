# -*- coding: UTF-8 -*-

import sys
sys.path.append("..")

import numpy as np
from exp_methods import *
from neural_network import *
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sequence, Sampler, Iterator
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLasso
import warnings
import random
from skimage.segmentation import slic


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


def prepare_cifar10_data(poison_ratio):
    batch_size = 1
    # Data
    print("==> Preparing data..")
    clear_train_data, clear_train_targets, pattern_train_data, pattern_train_targets, clear_test_data, clear_test_targets, pattern_test_data, pattern_test_targets = torch.load(
        '...')
    num_poison_test_data = 1000
    set_seed(1234)
    random_idx1 = list(range(num_poison_test_data))
    random.shuffle(random_idx1)
    set_seed(1234)
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

    clear_baseline_dataset = CIFAR10DATASET(clear_test_data[:num_poison_test_data], clear_test_targets[:num_poison_test_data], transform=transforms.Compose([
        transforms.GaussianBlur(11, 20),
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

    clear_test_loader = torch.utils.data.DataLoader(clear_test_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    sampler=test_sampler1)
    poison_test_loader = torch.utils.data.DataLoader(
        poison_test_dataset, batch_size=batch_size, shuffle=False,
        sampler=test_sampler1)

    clear_baseline_loader = torch.utils.data.DataLoader(clear_baseline_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        sampler=test_sampler2)
    poison_baseline_loader = torch.utils.data.DataLoader(
        poison_baseline_dataset, batch_size=batch_size, shuffle=False,
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
    return clear_test_loader, poison_test_loader, clear_baseline_loader, poison_baseline_loader, classes_name


def prepare_models(model_path_list):
    model_list = []
    for current_model_path in model_path_list:
        model = torch.load(current_model_path).to(device).eval()
        model_list.append(model)
    return model_list


def backdoor_pattern_coverage_test(coverage_rate, random_coverage_rate, method_score_list, true_topk, topk):
    for idx, score in enumerate(method_score_list):
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)
        current_topk = sort_score_position[:topk]
        random_topk = np.random.randint(0, 1024, size=[topk])
        coverage_rate[idx] += np.intersect1d(
            true_topk, current_topk).shape[0] / topk
        random_coverage_rate[idx] += np.intersect1d(
            true_topk, random_topk).shape[0] / topk
    return coverage_rate, random_coverage_rate


def method_comparison(clear_test_loader, poison_test_loader, clear_baseline_loader, poison_baseline_loader, model, model_name, method_name_list, t_mean, t_std, classes_name):
    coverage_rate = np.zeros(len(method_name_list))
    random_coverage_rate = np.zeros(len(method_name_list))
    # methods comparison
    for batch_idx, ((image, target), (baseline, _)) in enumerate(zip(poison_test_loader, poison_baseline_loader)):
        unnorm_img = image[0] * t_std + t_mean
        original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                      (1, 2, 0))
        input = image.to(device)
        baseline = baseline.to(device)

        # Saliency
        saliency_score = get_saliency_map_result(original_image, input, target,
                                                 model)

        # IG
        ig_score = get_ig_result(original_image, input, target, model)

        # SG
        sg_score = get_smoothgrad_result(original_image,
                                         input,
                                         target,
                                         model,
                                         stdevs=0.2)

        # SGSQ
        sgsq_score = get_smoothgradsq_result(original_image,
                                             input,
                                             target,
                                             model,
                                             stdevs=0.2)

        # SGVAR
        sgvar_score = get_smoothgradvar_result(original_image,
                                               input,
                                               target,
                                               model,
                                               stdevs=0.2)

        # SGIGSQ
        sgigsq_score = get_smoothgradigsq_result(original_image,
                                                 input,
                                                 target,
                                                 model,
                                                 stdevs=0.2)

        # DeepLIFT
        dl_score = get_deeplift_result(original_image,
                                       input,
                                       target,
                                       model,
                                       baselines=0)

        # Occlusion
        occlusion_score = get_occlusion_result(original_image,
                                               input,
                                               target,
                                               model,
                                               sliding_window_shapes=(3, 3, 3))

        # get superpixel
        img = np.transpose((image.numpy().squeeze()), (1, 2, 0))
        segments = slic(img,
                        n_segments=70,
                        compactness=0.1,
                        max_iter=10,
                        sigma=0)
        feature_mask = torch.Tensor(segments).long().to(device).unsqueeze(
            0).unsqueeze(0)

        # KS
        ks_score = get_ks_result(original_image,
                                 input,
                                 target,
                                 model,
                                 feature_mask=feature_mask,
                                 n_samples=500)

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

        method_score_list = [
            saliency_score, ig_score, sg_score, sgsq_score, sgvar_score, sgigsq_score, dl_score, occlusion_score, ks_score, lime_score
        ]

        reference_img = torch.zeros(32, 32)
        reference_img[21:25, 21:25] = 1

        reference_img = reference_img.numpy().reshape(-1)
        true_topk = np.nonzero(reference_img)[0]
        topk = true_topk.shape[0]

        coverage_rate, random_coverage_rate = backdoor_pattern_coverage_test(
            coverage_rate, random_coverage_rate, method_score_list, true_topk, topk)

    coverage_rate /= len(poison_test_loader)
    random_coverage_rate /= len(poison_test_loader)
    print('coverage_rate: {}'.format(
        coverage_rate))
    print('[Random] coverage_rate: {}'.format(
        random_coverage_rate))


if __name__ == "__main__":
    model_path_list = ['...']
    model_name_list = ['...']
    method_name_list = [
        'Saliency', 'IG', 'SmoothGrad', 'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime'
    ]
    poison_ratio = 0.05
    # load data, models
    clear_test_loader, poison_test_loader, clear_baseline_loader, poison_baseline_loader, classes_name = prepare_cifar10_data(
        poison_ratio)
    cifar10_model_list = prepare_models(model_path_list)

    std = (0.4914, 0.4822, 0.4465)
    mu = (0.2023, 0.1994, 0.2010)
    t_mean = torch.FloatTensor(mu).view(3, 1, 1).expand(3, 32, 32)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 32, 32)

    for idx, model in enumerate(cifar10_model_list):
        method_comparison(clear_test_loader, poison_test_loader, clear_baseline_loader, poison_baseline_loader, model,
                          model_name_list[idx], method_name_list, t_mean, t_std, classes_name)
