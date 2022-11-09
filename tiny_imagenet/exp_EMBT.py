# -*- coding: UTF-8 -*-

import torch
import sys

sys.path.append("..")
import os
from torchvision import datasets
from torchvision import transforms as T
from PIL import Image

from exp_methods import *
import numpy as np
from skimage.segmentation import slic
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from torch.utils.data.sampler import Sequence, Sampler, Iterator
import random
import warnings
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    # Data
    print("==> Preparing data..")
    n_train = 1000
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
    poison_test_loader_1 = DataLoader(datasets.ImageFolder(
        './backdoor_data/val_pattern/', transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])), batch_size=1, shuffle=False, num_workers=32, sampler=val_sampler1)
    poison_test_loader_64 = DataLoader(datasets.ImageFolder(
        './backdoor_data/val_pattern/', transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])), batch_size=32, shuffle=False, num_workers=32)
    poison_baseline_loader = DataLoader(datasets.ImageFolder(
        './backdoor_data/val_pattern/', transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.GaussianBlur(17, 30),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])), batch_size=1, shuffle=False, num_workers=32, sampler=val_sampler2)

    return poison_test_loader_1, poison_test_loader_64, poison_baseline_loader


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
        random_topk = np.random.randint(0, 50176, size=[topk])
        coverage_rate[idx] += np.intersect1d(
            true_topk, current_topk).shape[0] / topk
        random_coverage_rate[idx] += np.intersect1d(
            true_topk, random_topk).shape[0] / topk
    return coverage_rate, random_coverage_rate


if __name__ == "__main__":
    classes_name = {}
    with open("./tiny-imagenet-200/words.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split('\t', 1)
            classes_name[line[0]] = line[1]
    model_path_list = ['...']
    method_name_list = [
        'Saliency', 'IG', 'SmoothGrad', 'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime'
    ]
    start = time.clock()

    # load data, models
    poison_test_loader_1, poison_test_loader_64, poison_baseline_loader = prepare_tinyimagenet_data()  # batch_size=1

    tinyimagenet_model_list = prepare_models(model_path_list)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 224, 224)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 224, 224)

    num_explain = 50

    # Reference Background Truth
    reference_img = np.zeros((64, 64), dtype=np.float)
    reference_img[51:58, 51:58] = 1
    reference_img = Image.fromarray(reference_img)
    resize = T.Resize([256, 256])
    reference_img = resize(reference_img)
    center_crop = T.CenterCrop([224, 224])
    reference_img = center_crop(reference_img)
    reference_img = np.array(reference_img).reshape(-1)
    true_topk = np.nonzero(reference_img)[0]
    topk = true_topk.shape[0]

    pearson_coff = np.zeros(len(method_name_list))
    sum_coverage_rate = np.zeros((len(method_name_list), len(model_path_list)))
    sum_random_coverage_rate = np.zeros(
        (len(method_name_list), len(model_path_list)))
    sum_backdoor_confidence_list = np.zeros(len(tinyimagenet_model_list))
    for batch_idx_1, ((data, target), (baseline, target2)) in enumerate(zip(poison_test_loader_1, poison_baseline_loader)):
        backdoor_confidence_list = np.zeros(len(tinyimagenet_model_list))
        coverage_rate = np.zeros((len(method_name_list), len(model_path_list)))
        random_coverage_rate = np.zeros(
            (len(method_name_list), len(model_path_list)))
        if batch_idx_1 == num_explain:
            break
        unnorm_img = data[0] * t_std + t_mean
        original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                      (1, 2, 0))
        input, target = data.to(device), target.to(device)
        baseline = baseline.to(device)

        explanatory_result = np.zeros(
            (len(tinyimagenet_model_list), len(method_name_list), 224, 224))
        for model_idx, model in enumerate(tinyimagenet_model_list):
            output = F.softmax(model(input), dim=1)
            backdoor_confidence_list[model_idx] = output[0][target]
            sum_backdoor_confidence_list[model_idx] += backdoor_confidence_list[model_idx]
            # get explanatory result
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
            img = np.transpose((data.numpy().squeeze()), (1, 2, 0))
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
            coverage_rate[:, model_idx], random_coverage_rate[:, model_idx] = backdoor_pattern_coverage_test(
                coverage_rate[:, model_idx], random_coverage_rate[:, model_idx], method_score_list, true_topk, topk)

            sum_coverage_rate[:, model_idx] += coverage_rate[:, model_idx]
            sum_random_coverage_rate[:,
                                     model_idx] += random_coverage_rate[:, model_idx]

        for method_idx in range(len(method_score_list)):
            if (coverage_rate[method_idx] == coverage_rate[method_idx, 0]).all():
                continue
            pearson_coff[method_idx] += np.corrcoef(
                backdoor_confidence_list, coverage_rate[method_idx])[0, 1]

    end = time.clock()
    print('avg pearson: {}, backdoor_acc: {}, avg_coverage: {}, random_coverage: {}'.format(
        pearson_coff / num_explain, sum_backdoor_confidence_list / num_explain, sum_coverage_rate / num_explain, sum_random_coverage_rate / num_explain))
    print('time: {}'.format(end - start))
