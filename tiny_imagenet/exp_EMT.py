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
import time

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
    val_loader_1 = DataLoader(datasets.ImageFolder(
        VALID_DIR, transform=valid_transform), batch_size=1, shuffle=False, num_workers=32, sampler=val_sampler1)
    val_loader_64 = DataLoader(datasets.ImageFolder(
        VALID_DIR, transform=valid_transform), batch_size=64, shuffle=False, num_workers=32)
    baseline_loader = DataLoader(datasets.ImageFolder(
        VALID_DIR, transform=baseline_transform), batch_size=1, shuffle=False, num_workers=32, sampler=val_sampler2)

    return val_loader_1, val_loader_64, baseline_loader


def prepare_models(model_path_list):
    model_list = []
    for current_model_path in model_path_list:
        model = torch.load(current_model_path).to(device).eval()
        model_list.append(model)
    return model_list


if __name__ == "__main__":
    start = time.clock()
    classes_name = {}
    with open("./tiny-imagenet-200/words.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split('\t', 1)
            classes_name[line[0]] = line[1]
    model_path_list = ['...']

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 224, 224)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 224, 224)

    # load data, models
    test_loader_1, test_loader_64, baseline_loader = prepare_tinyimagenet_data()
    tinyimagenet_model_list = prepare_models(model_path_list)

    num_explain = 50
    num_method = 10
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    sum_delta_loss = np.zeros(len(tinyimagenet_model_list) - 1)
    sum_delta_explanatory = np.zeros(
        (num_method, len(tinyimagenet_model_list) - 1))
    pearson_coff = np.zeros(num_method)

    loss_result = np.zeros(len(tinyimagenet_model_list))
    # get loss
    for model_idx, model in enumerate(tinyimagenet_model_list):
        test_loss = 0
        with torch.no_grad():
            for batch_idx_64, (inputs, targets) in enumerate(test_loader_64):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        loss_result[model_idx] = test_loss

    for batch_idx_1, ((data, target), (baseline, target2)) in enumerate(zip(test_loader_1, baseline_loader)):
        if batch_idx_1 == num_explain:
            break

        unnorm_img = data[0] * t_std + t_mean
        original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                      (1, 2, 0))
        input = data.to(device)
        baseline = baseline.to(device)

        explanatory_result = np.zeros(
            (len(tinyimagenet_model_list), num_method, int(0.1 * 224 * 224)))
        for model_idx, model in enumerate(tinyimagenet_model_list):
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
            # get top k
            topk = int(num_feature * 0.1)
            sort_all_score = np.zeros((num_method, topk))
            for idx, score in enumerate(method_score_list):
                flatten_score = score.reshape(-1)
                sort_score_position = np.argsort(-flatten_score)[:topk]
                sort_all_score[idx] = sort_score_position
            explanatory_result[model_idx] = sort_all_score

        # compute sim
        sim_explanetory_result = np.zeros(
            (num_method, len(tinyimagenet_model_list) - 1))
        for model_idx in range(explanatory_result.shape[0] - 1):
            for method_idx in range(explanatory_result.shape[1]):
                sim_explanetory_result[method_idx, model_idx] += 1 - np.intersect1d(
                    explanatory_result[model_idx, method_idx], explanatory_result[model_idx + 1, method_idx]).shape[0] / explanatory_result.shape[-1]

        # if cov == nan then continue
        continue_flag = False
        for idx, current_sim_explanetory_result in enumerate(sim_explanetory_result):
            if (current_sim_explanetory_result == current_sim_explanetory_result[0]).all():
                continue_flag = True
                break

        if continue_flag:
            continue

        # compute delta
        delta_loss_list = np.zeros(len(loss_result) - 1)
        loss_result = np.array(loss_result)
        for idx in range(len(loss_result) - 1):
            delta_loss_list[idx] = loss_result[idx] - loss_result[idx + 1]

        sum_delta_loss += delta_loss_list
        sum_delta_explanatory += sim_explanetory_result
        # compute pearson coef
        for method_idx in range(sim_explanetory_result.shape[0]):
            pearson_coff[method_idx] += np.corrcoef(
                delta_loss_list, sim_explanetory_result[method_idx])[0, 1]
    avg_pearson_coff = pearson_coff / num_explain
    avg_delta_loss = sum_delta_loss / num_explain
    avg_delta_explanatory = sum_delta_explanatory / num_explain
    end = time.clock()
    print('avg pearson coff: {}, avg delta loss: {}, avg delta explanatory: {}'.format(
        avg_pearson_coff, avg_delta_loss, avg_delta_explanatory))
    print('time: {}'.format(end - start))
