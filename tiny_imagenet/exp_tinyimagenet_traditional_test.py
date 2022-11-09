# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import os
import torch
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader
from exp_methods import *
import numpy as np
from skimage.segmentation import slic
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from resnet_models import *
from torch.utils.data.sampler import Sequence, Sampler, Iterator
import random
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
black_pixel_val = (torch.zeros(
    3) - torch.Tensor((0.485, 0.456, 0.406))) / torch.Tensor((0.229, 0.224, 0.225))


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


def prepare_tinyimagenet_data():
    batch_size = 1
    # Data
    print("==> Preparing data..")
    n_train = 10000
    set_seed(1234)
    random_idx1 = list(range(n_train))
    random.shuffle(random_idx1)
    set_seed(1234)
    random_idx2 = list(range(n_train))
    random.shuffle(random_idx2)
    val_sampler1 = CustomSampler(
        random_idx1)
    val_sampler2 = CustomSampler(
        random_idx2)

    # Define training and validation data paths
    DATA_DIR = 'tiny-imagenet-200'
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    valid_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    baseline_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.GaussianBlur(17, 30),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    val_loader = DataLoader(datasets.ImageFolder(
        VALID_DIR, transform=valid_transform), batch_size=batch_size, shuffle=False, num_workers=32, sampler=val_sampler1)
    baseline_loader = DataLoader(datasets.ImageFolder(
        VALID_DIR, transform=baseline_transform), batch_size=batch_size, shuffle=False, num_workers=32, sampler=val_sampler2)

    return val_loader, baseline_loader


def feature_reduction_test(current_data, target, model, reduction_rate, random_reduction_rate, method_score_list, topk, num_feature):
    origin_data = current_data.clone().detach()
    origin_predict = model(origin_data.to(device))
    for idx, score in enumerate(method_score_list):
        reduction_data = current_data.clone().detach()
        random_data = current_data.clone().detach()
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)[:topk]
        current_topkx = sort_score_position // 224
        current_topky = sort_score_position % 224
        for x, y in zip(current_topkx, current_topky):
            reduction_data[:, :, x, y] = black_pixel_val

        random_topk = np.random.randint(0, 50176, size=[topk])
        current_topkx = random_topk // 224
        current_topky = random_topk % 224
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
        current_topkx = sort_score_position // 224
        current_topky = sort_score_position % 224
        for x, y in zip(current_topkx, current_topky):
            synthesis_data[:, :, x, y] = origin_data[:, :, x, y]

        random_topk = np.random.randint(0, 50176, size=[topk])
        current_topkx = random_topk // 224
        current_topky = random_topk % 224
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
        current_topkx = sort_score_position // 224
        current_topky = sort_score_position % 224
        for x, y in zip(current_topkx, current_topky):
            augmentation_data[:, :, x, y] = current_data[:, :, x, y]

        random_topk = np.random.randint(0, 50176, size=[topk])
        current_topkx = random_topk // 224
        current_topky = random_topk % 224
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


def method_comparison(test_loader, baseline_loader, model, model_name, method_name_list, num_explain, t_mean, t_std, classes_name):
    reduction_rate = np.zeros(len(method_name_list))
    systhesis_rate = np.zeros(len(method_name_list))
    augmentation_rate = np.zeros(len(method_name_list))
    random_reduction_rate = np.zeros(len(method_name_list))
    random_systhesis_rate = np.zeros(len(method_name_list))
    random_augmentation_rate = np.zeros(len(method_name_list))
    # methods comparison
    counter = 0
    for batch_idx, ((image, target), (baseline, targets2)) in enumerate(zip(val_loader, baseline_loader)):
        if counter == num_explain:
            break
        unnorm_img = image[0] * t_std + t_mean
        original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                      (1, 2, 0))
        input = image.to(device)
        baseline = baseline.to(device)
        output = model(input)
        if torch.argmax(output).cpu() != target:
            continue
        counter += 1

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

        # SGVAR
        sgvar_score = get_smoothgradvar_result(original_image,
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
                                               sliding_window_shapes=(3, 6, 6))

        # get superpixel
        img = np.transpose((image.numpy().squeeze()), (1, 2, 0))
        segments = slic(img,
                        n_segments=100,
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
            interpretable_model=SkLearnLasso(alpha=0.08),
            feature_mask=feature_mask,
            similarity_func=exp_eucl_distance,
            n_samples=500)

        method_score_list = [
            saliency_score, ig_score, sg_score, sgsq_score, sgvar_score, sgigsq_score, dl_score, occlusion_score, ks_score, lime_score
        ]

        num_feature = saliency_score.reshape(-1).shape[0]
        topk = int(num_feature * 0.1)

        reduction_rate, random_reduction_rate = feature_reduction_test(
            image, target, model, reduction_rate, random_reduction_rate, method_score_list, topk, num_feature)
        systhesis_rate, random_systhesis_rate = feature_synthesis_test(
            image, target, model, systhesis_rate, random_systhesis_rate, method_score_list, topk, num_feature)
        augmentation_rate, random_augmentation_rate = feature_augmentation_test(
            test_loader, image, target, model, augmentation_rate, random_augmentation_rate, method_score_list, topk, num_feature)
    reduction_rate /= num_explain
    systhesis_rate /= num_explain
    augmentation_rate /= num_explain
    random_reduction_rate /= num_explain
    random_systhesis_rate /= num_explain
    random_augmentation_rate /= num_explain
    print('reduction_rate: {}, systhesis_rate: {}, augmentation_rate: {}'.format(
        reduction_rate, systhesis_rate, augmentation_rate))
    print('[Random] reduction_rate: {}, systhesis_rate: {}, augmentation_rate: {}'.format(
        random_reduction_rate, random_systhesis_rate, random_augmentation_rate))


if __name__ == "__main__":
    torch.manual_seed(1234)
    classes_name = {}
    with open("./tiny-imagenet-200/words.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split('\t', 1)
            classes_name[line[0]] = line[1]
    model_name_list = ['...']
    method_name_list = [
        'Saliency', 'IG', 'SmoothGrad', 'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime'
    ]

    # load data, models
    val_loader, baseline_loader = prepare_tinyimagenet_data()  # batch_size=1
    resnet18_model = torch.load('...').eval().to(device)
    imagenet_model_list = [resnet18_model]

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 224, 224)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 224, 224)

    num_explain = 100

    for idx, model in enumerate(imagenet_model_list):
        method_comparison(val_loader, baseline_loader, model,
                          model_name_list[idx], method_name_list, num_explain, t_mean, t_std, classes_name)
