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

warnings.filterwarnings('ignore')


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
        '...'
    )

    concat_pattern_data = list(zip(clear_test_data, clear_test_targets))
    random.shuffle(concat_pattern_data)
    clear_test_data, clear_test_targets = zip(*concat_pattern_data)

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


def img_process(data):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 32, 32)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 32, 32)
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
        random_topk = np.random.randint(0, 1024, size=[topk])
        coverage_rate[idx] += np.intersect1d(
            true_topk, current_topk).shape[0] / topk
        random_coverage_rate[idx] += np.intersect1d(
            true_topk, random_topk).shape[0] / topk
    return coverage_rate, random_coverage_rate


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.clock()
    poison_ratio = 0.01
    num_method = 10
    num_explain = 100
    clear_test_loader, poison_test_loader, clear_baseline_loader, poison_baseline_loader, classes_name = prepare_cifar10_data(
        poison_ratio)
    model = torch.load('...').to(device).eval()
    pattern_digital_data = torch.zeros((32, 32, 3))

    pattern_digital_data[26:30, 26:30] = 1
    pattern_digital_data[27:30, 26] = 0
    pattern_digital_data[27:30, 29] = 0

    pattern_digital_data[4:8, 4:8] = 1
    pattern_digital_data[5:8, 4] = 0
    pattern_digital_data[5:8, 7] = 0

    pattern_digital_data[26:30, 4:8] = 1
    pattern_digital_data[27:30, 4] = 0
    pattern_digital_data[27:30, 7] = 0

    pattern_digital_data[4:8, 26:30] = 1
    pattern_digital_data[5:8, 26] = 0
    pattern_digital_data[5:8, 29] = 0
    pattern_digital_data = (pattern_digital_data - torch.Tensor(
        (0.4914, 0.4822, 0.4465))) / torch.Tensor((0.2023, 0.1994, 0.2010))
    pattern_digital_data = pattern_digital_data.permute((2, 0, 1))

    pattern_position_data = torch.zeros((32, 32))
    pattern_position_data[26:30, 26:30] = 1
    pattern_position_data[4:8, 4:8] = 1
    pattern_position_data[26:30, 4:8] = 1
    pattern_position_data[4:8, 26:30] = 1
    pos_pattern = torch.nonzero(pattern_position_data)
    pattern_img = pattern_position_data.numpy().reshape(-1)
    true_topk = np.nonzero(pattern_img)[0]
    portion = np.arange(1, 11) * 0.1
    # portion = [1.0]
    sum_confidence = np.zeros(10)
    sum_coverage = np.zeros((num_method, 10))
    pearson_coff = np.zeros(num_method)
    random_pearson_coff = np.zeros(num_method)
    counter = 0
    for batch_idx, ((data, target), (baseline, _)) in enumerate(zip(clear_test_loader, clear_baseline_loader)):
        if int(target) == 0:
            continue
        if counter == num_explain:
            break
        counter += 1
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
            num_pattern_feat = int(16 * current_portion)
            rand_pos_pattern = np.random.choice(
                16, num_pattern_feat, replace=False)
            partial_backdoor_pos.append(true_topk[rand_pos_pattern])
            partial_topk.append(len(rand_pos_pattern))
            tmp_data = clear_data.clone().detach()
            for current_rand_pos_pattern in rand_pos_pattern:
                current_pos_pattern = pos_pattern[current_rand_pos_pattern]
                pos_x = current_pos_pattern[0]
                pos_y = current_pos_pattern[1]
                tmp_data[:, :, pos_x,
                         pos_y] = pattern_digital_data[:, pos_x, pos_y]
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
                                                   sliding_window_shapes=(3, 3, 3))

            # get superpixel
            img = np.transpose((data.numpy().squeeze()), (1, 2, 0))
            segments = slic(img,
                            n_segments=70,
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
                interpretable_model=SkLearnLasso(alpha=0.05),
                feature_mask=feature_mask,
                similarity_func=exp_eucl_distance,
                n_samples=500)

            method_score_list = [
                saliency_score, ig_score, sg_score, sgsq_score, sgvar_score, sgigsq_score, dl_score, occlusion_score, ks_score, lime_score
            ]
            num_feature = saliency_score.reshape(-1).shape[0]

            coverage_rate = np.zeros(len(method_score_list))
            random_coverage_rate = np.zeros(len(method_score_list))
            coverage_rate, random_coverage_rate = backdoor_pattern_coverage_test(
                coverage_rate, random_coverage_rate, method_score_list, partial_backdoor_pos[test_data_idx], int(num_feature * 0.1))
            test_coverage_rate.append(coverage_rate)
            test_random_coverage_rate.append(random_coverage_rate)

        # compute tendency correlation
        test_confidence = np.array(test_confidence)
        sum_confidence += test_confidence
        test_coverage_rate = np.array(test_coverage_rate).transpose()
        sum_coverage += test_coverage_rate
        test_random_coverage_rate = np.array(
            test_random_coverage_rate).transpose()

        # if cov == nan then continue
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

    avg_pearson_coff = pearson_coff / counter
    avg_random_pearson_coff = random_pearson_coff / counter
    avg_confidence = sum_confidence / counter
    avg_coverage = sum_coverage / counter
    end = time.clock()
    print('counter: {}, avg peasron: {}, random: {}, avg confidence: {}, avg coverage: {}'.format(
        counter, avg_pearson_coff, avg_random_pearson_coff, avg_confidence, avg_coverage))
    print('time: {}'.format(end - start))
