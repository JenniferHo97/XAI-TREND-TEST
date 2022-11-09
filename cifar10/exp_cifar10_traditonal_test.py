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
import random
import torch
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

    test_data = torchvision.datasets.CIFAR10(root="./data",
                                             train=False,
                                             download=True,
                                             transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
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
    return test_loader, baseline_loader, classes_name


def prepare_models(model_path_list):
    model_list = []
    for current_model_path in model_path_list:
        model = torch.load(current_model_path).to(device).eval()
        model_list.append(model)
    return model_list


def feature_reduction_test(current_data, target, model, reduction_rate, random_reduction_rate, method_score_list, topk, num_feature):
    origin_data = current_data.clone().detach()
    origin_predict = model(origin_data.to(device))
    for idx, score in enumerate(method_score_list):
        reduction_data = current_data.clone().detach()
        random_data = current_data.clone().detach()
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)[:topk]
        current_topkx = sort_score_position // 32
        current_topky = sort_score_position % 32
        for x, y in zip(current_topkx, current_topky):
            reduction_data[:, :, x, y] = black_pixel_val

        random_topk = np.random.randint(0, 1024, size=[topk])
        current_topkx = random_topk // 32
        current_topky = random_topk % 32
        for x, y in zip(current_topkx, current_topky):
            random_data[:, :, x, y] = black_pixel_val

        reduction_predict = model(reduction_data.to(device))

        reduction_rate[idx] += torch.softmax(
            origin_predict, -1)[0, target] - torch.softmax(reduction_predict, -1)[0, target]

        random_predict = model(random_data.to(device))
        random_reduction_rate[idx] += torch.softmax(
            origin_predict, -1)[0, target] - torch.softmax(random_predict, -1)[0, target]
    return reduction_rate, random_reduction_rate


def feature_synthesis_test(current_data, target, model, synthesis_rate, random_synthesis_rate, method_score_list, topk, num_feature):
    origin_data = current_data.clone().detach()
    for idx, score in enumerate(method_score_list):
        synthesis_data = torch.zeros_like(current_data)
        zero_data = torch.zeros_like(current_data)
        random_data = torch.zeros_like(current_data)
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)[:topk]
        current_topkx = sort_score_position // 32
        current_topky = sort_score_position % 32
        for x, y in zip(current_topkx, current_topky):
            synthesis_data[:, :, x, y] = origin_data[:, :, x, y]

        random_topk = np.random.randint(0, 1024, size=[topk])
        current_topkx = random_topk // 32
        current_topky = random_topk % 32
        for x, y in zip(current_topkx, current_topky):
            random_data[:, :, x, y] = origin_data[:, :, x, y]

        zero_predict = model(zero_data.to(device))
        synthesis_predict = model(synthesis_data.to(device))
        synthesis_rate[idx] += torch.softmax(
            synthesis_predict, -1)[0, target] - torch.softmax(zero_predict, -1)[0, target]

        random_predict = model(random_data.to(device))
        random_synthesis_rate[idx] += torch.softmax(
            random_predict, -1)[0, target] - torch.softmax(zero_predict, -1)[0, target]
    return synthesis_rate, random_synthesis_rate


def feature_augmentation_test(test_loader, current_data, target, model, augmentation_rate, random_augmentation_rate, method_score_list, topk, num_feature):
    for idx, (data, label) in enumerate(test_loader):
        origin_data = data
        if target != label:
            break
    for idx, score in enumerate(method_score_list):
        random_data = origin_data.clone().detach()
        augmentation_data = origin_data.clone().detach()
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)[:topk]
        current_topkx = sort_score_position // 32
        current_topky = sort_score_position % 32
        for x, y in zip(current_topkx, current_topky):
            augmentation_data[:, :, x, y] = current_data[:, :, x, y]

        random_topk = np.random.randint(0, 1024, size=[topk])
        current_topkx = random_topk // 32
        current_topky = random_topk % 32
        for x, y in zip(current_topkx, current_topky):
            random_data[:, :, x, y] = current_data[:, :, x, y]

        origin_predict = model(origin_data.to(device))
        augmentation_predict = model(augmentation_data.to(device))
        augmentation_rate[idx] += torch.softmax(
            augmentation_predict, -1)[0, target] - torch.softmax(origin_predict, -1)[0, target]

        random_predict = model(random_data.to(device))
        random_augmentation_rate[idx] += torch.softmax(
            random_predict, -1)[0, target] - torch.softmax(origin_predict, -1)[0, target]
    return augmentation_rate, random_augmentation_rate


def method_comparison(test_loader, baseline_loader, model, model_name, method_name_list, t_mean, t_std, classes_name, num_explain):
    reduction_rate = np.zeros(len(method_name_list))
    systhesis_rate = np.zeros(len(method_name_list))
    augmentation_rate = np.zeros(len(method_name_list))
    random_reduction_rate = np.zeros(len(method_name_list))
    random_systhesis_rate = np.zeros(len(method_name_list))
    random_augmentation_rate = np.zeros(len(method_name_list))
    # methods comparison
    for batch_idx, ((image, target), (baseline, target2)) in enumerate(zip(test_loader, baseline_loader)):
        if batch_idx == num_explain:
            break
        unnorm_img = image[0] * t_std + t_mean
        original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                      (1, 2, 0))
        input = image.to(device)
        baseline = baseline.to(device)
        output = model(input)
        if torch.argmax(output).cpu() != target:
            continue

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

        num_feature = saliency_score.size
        topk = int(num_feature * 0.10)

        reduction_rate, random_reduction_rate = feature_reduction_test(
            image, target, model, reduction_rate, random_reduction_rate, method_score_list, topk, num_feature)
        systhesis_rate, random_systhesis_rate = feature_synthesis_test(
            image, target, model, systhesis_rate, random_systhesis_rate, method_score_list, topk, num_feature)
        augmentation_rate, random_augmentation_rate = feature_augmentation_test(test_loader, image, target,
                                                                                model, augmentation_rate, random_augmentation_rate, method_score_list, topk, num_feature)
    reduction_rate /= num_explain
    systhesis_rate /= num_explain
    augmentation_rate /= num_explain
    random_reduction_rate /= num_explain
    random_systhesis_rate /= num_explain
    random_augmentation_rate /= num_explain
    print(model_name)
    print('reduction_rate: {}, systhesis_rate: {}, augmentation_rate: {}'.format(
        reduction_rate, systhesis_rate, augmentation_rate))
    print('[Random] reduction_rate: {}, systhesis_rate: {}, augmentation_rate: {}'.format(
        random_reduction_rate, random_systhesis_rate, random_augmentation_rate))


if __name__ == "__main__":
    start = time.clock()
    model_path_list = ['...']
    model_name_list = ['...']
    method_name_list = [
        'Saliency', 'IG', 'SmoothGrad', 'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime'
    ]
    # load data, models
    test_loader, baseline_loader, classes_name = prepare_cifar10_data()
    cifar10_model_list = prepare_models(model_path_list)

    mu = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    t_mean = torch.FloatTensor(mu).view(3, 1, 1).expand(3, 32, 32)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 32, 32)

    num_explain = 500

    for idx, model in enumerate(cifar10_model_list):
        method_comparison(test_loader, baseline_loader, model,
                          model_name_list[idx], method_name_list, t_mean, t_std, classes_name, num_explain)
    end = time.clock()
    print('time: {}'.format(end - start))
