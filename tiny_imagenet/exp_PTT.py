# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import os
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLasso
from skimage.segmentation import slic
import numpy as np
from exp_methods import *
from resnet_models import *
from torchvision import transforms as T
from torchvision import datasets
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sequence, Sampler, Iterator
import warnings
import random
import time

warnings.filterwarnings('ignore')


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
    n_train = 1000
    set_seed(1234)
    random_idx1 = list(range(n_train))
    random.shuffle(random_idx1)
    set_seed(1234)
    random_idx2 = list(range(n_train))
    random.shuffle(random_idx2)
    set_seed(1234)
    random_idx3 = list(range(n_train))
    random.shuffle(random_idx3)
    set_seed(1234)
    random_idx4 = list(range(n_train))
    random.shuffle(random_idx4)
    val_sampler1 = CustomSampler(
        random_idx1)
    val_sampler2 = CustomSampler(
        random_idx2)
    val_sampler3 = CustomSampler(
        random_idx3)
    val_sampler4 = CustomSampler(
        random_idx4)

    # Define training and validation data paths
    DATA_DIR = 'tiny-imagenet-200'
    VALID_DIR = os.path.join(DATA_DIR, 'val')

    clear_test_loader = DataLoader(datasets.ImageFolder(
        './backdoor_data/val_clear/', transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])), batch_size=batch_size, shuffle=False, num_workers=32, sampler=val_sampler1)
    clear_baseline_loader = DataLoader(datasets.ImageFolder(
        './backdoor_data/val_clear/', transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.GaussianBlur(17, 30),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])), batch_size=batch_size, shuffle=False, num_workers=32, sampler=val_sampler2)
    poison_test_loader = DataLoader(datasets.ImageFolder(
        './backdoor_data/val_pattern/', transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])), batch_size=batch_size, shuffle=False, num_workers=32, sampler=val_sampler1)
    poison_baseline_loader = DataLoader(datasets.ImageFolder(
        './backdoor_data/val_pattern/', transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.GaussianBlur(17, 30),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])), batch_size=batch_size, shuffle=False, num_workers=32, sampler=val_sampler2)

    return clear_test_loader, clear_baseline_loader, poison_test_loader, poison_baseline_loader


def img_process(data):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 224, 224)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 224, 224)
    img = data.cpu().clone().squeeze(0)
    img = img * t_std + t_mean
    img = img.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    return img


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


if __name__ == '__main__':
    start = time.clock()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    poison_ratio = 0.05
    clear_test_loader, clear_baseline_loader, poison_test_loader, poison_baseline_loader = prepare_tinyimagenet_data()
    model = torch.load('...').to(device).eval()

    num_method = 10
    num_explain = 100

    reference_img = np.zeros((64, 64), dtype=np.float)
    reference_img[51:58, 51:58] = 1
    reference_img = Image.fromarray(reference_img)
    resize = T.Resize([256, 256])
    reference_img = resize(reference_img)
    center_crop = T.CenterCrop([224, 224])
    reference_img = center_crop(reference_img)
    to_tensor = T.ToTensor()
    reference_img = to_tensor(reference_img)
    pos_pattern = torch.nonzero(reference_img)
    pattern_img = np.array(reference_img).reshape(-1)
    true_topk = np.nonzero(pattern_img)[0]
    portion = np.arange(1, 11) * 0.1
    sum_confidence = np.zeros(10)
    sum_coverage = np.zeros((num_method, 10))
    pearson_coff = np.zeros(num_method)
    random_pearson_coff = np.zeros(num_method)
    for batch_idx, ((data, target), (baseline, _), (data2, target2), (baseline2, _)) in enumerate(zip(clear_test_loader, clear_baseline_loader, poison_test_loader, poison_baseline_loader)):
        if batch_idx == num_explain:
            break

        target = 0
        original_image = np.transpose((data[0].cpu().detach().numpy()),
                                      (1, 2, 0))
        clear_data = data.to(device)
        baseline = baseline.to(device)

        # generate dynamic data
        test_data = []
        partial_backdoor_pos = []
        partial_topk = []
        for current_portion in portion:
            num_pattern_feat = int(true_topk.shape[0] * current_portion)
            rand_pos_pattern = np.random.choice(
                true_topk.shape[0], num_pattern_feat, replace=False)
            partial_backdoor_pos.append(true_topk[rand_pos_pattern])
            partial_topk.append(len(rand_pos_pattern))
            tmp_data = clear_data.clone().detach()
            for current_rand_pos_pattern in rand_pos_pattern:
                current_pos_pattern = pos_pattern[current_rand_pos_pattern]
                pos_x = current_pos_pattern[1]
                pos_y = current_pos_pattern[2]
                white_pixel_val = (torch.ones(
                    3) - torch.Tensor((0.485, 0.456, 0.406))) / torch.Tensor((0.229, 0.224, 0.225))
                tmp_data[:, :, pos_x, pos_y] = data2[:, :, pos_x, pos_y]
            test_data.append(tmp_data)

        # get confidences and explantory results
        test_confidence = []
        test_coverage_rate = []
        test_random_coverage_rate = []
        for test_data_idx, current_data in enumerate(test_data):
            tmp_confidence = torch.softmax(model(current_data), -1)[0, 0]
            test_confidence.append(tmp_confidence.cpu().detach().numpy())

            # Saliency
            saliency_score = get_saliency_map_result(original_image, current_data, target,
                                                     model)

            # IG
            ig_score = get_ig_result(
                original_image, current_data, target, model)
            # SG
            sg_score = get_smoothgrad_result(original_image,
                                             current_data,
                                             target,
                                             model,
                                             stdevs=0.2)
            # SGVAR
            sgvar_score = get_smoothgradvar_result(original_image,
                                                   current_data,
                                                   target,
                                                   model,
                                                   stdevs=0.2)

            # SGSQ
            sgsq_score = get_smoothgradsq_result(original_image,
                                                 current_data,
                                                 target,
                                                 model,
                                                 stdevs=0.2)

            # SGIGSQ
            sgigsq_score = get_smoothgradigsq_result(original_image,
                                                     current_data,
                                                     target,
                                                     model,
                                                     stdevs=0.2)

            # DeepLIFT
            dl_score = get_deeplift_result(original_image,
                                           current_data,
                                           target,
                                           model,
                                           baselines=0)

            # Occlusion
            occlusion_score = get_occlusion_result(original_image,
                                                   current_data,
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
                                     current_data,
                                     target,
                                     model,
                                     feature_mask=feature_mask,
                                     n_samples=500)

            # Lime
            exp_eucl_distance = get_exp_kernel_similarity_function(
                'euclidean', kernel_width=1000)
            lime_score = get_lime_result(
                original_image,
                current_data,
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
            topk = true_topk.shape[0]

            coverage_rate = np.zeros(len(method_score_list))
            random_coverage_rate = np.zeros(len(method_score_list))
            coverage_rate, random_coverage_rate = backdoor_pattern_coverage_test(
                coverage_rate, random_coverage_rate, method_score_list, partial_backdoor_pos[test_data_idx], partial_topk[test_data_idx])
            test_coverage_rate.append(coverage_rate)
            test_random_coverage_rate.append(random_coverage_rate)

        # compute tendency correlation
        test_confidence = np.array(test_confidence)
        sum_confidence += test_confidence
        test_coverage_rate = np.array(test_coverage_rate).transpose()
        sum_coverage += test_coverage_rate
        test_random_coverage_rate = np.array(
            test_random_coverage_rate).transpose()

        continue_flag = False
        for idx, (current_test_coverage_rate, current_random_coverage_rate) in enumerate(zip(test_coverage_rate, test_random_coverage_rate)):
            if (current_test_coverage_rate == current_test_coverage_rate[0]).all() or (test_random_coverage_rate == test_random_coverage_rate[0]).all():
                continue_flag = True
                break

        if continue_flag:
            continue

        for idx, (current_test_coverage_rate, current_random_coverage_rate) in enumerate(zip(test_coverage_rate, test_random_coverage_rate)):
            pearson_coff[idx] += (np.corrcoef(
                test_confidence, current_test_coverage_rate))[0, 1]
            random_pearson_coff[idx] += (np.corrcoef(
                test_confidence, current_random_coverage_rate))[0, 1]
    avg_pearson_coff = pearson_coff / num_explain
    avg_random_pearson_coff = random_pearson_coff / num_explain
    avg_confidence = sum_confidence / num_explain
    avg_coverage = sum_coverage / num_explain
    end = time.clock()
    print('avg pearson: {}, random: {}, avg confidence: {}, avg coverage: {}'.format(
        avg_pearson_coff, avg_random_pearson_coff, avg_confidence, avg_coverage))
    print('time: {}'.format(end - start))
