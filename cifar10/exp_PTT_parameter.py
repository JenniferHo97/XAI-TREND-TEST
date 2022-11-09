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
        '...')

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
    num_explain = 100
    clear_test_loader, poison_test_loader, clear_baseline_loader, poison_baseline_loader, classes_name = prepare_cifar10_data(
        poison_ratio)
    model = torch.load('...').to(device).eval()
    pattern_digital_data = torch.zeros((32, 32, 3))
    pattern_digital_data[21:25, 21: 25] = 1
    pattern_digital_data = (pattern_digital_data - torch.Tensor(
        (0.4914, 0.4822, 0.4465))) / torch.Tensor((0.2023, 0.1994, 0.2010))
    pattern_digital_data = pattern_digital_data.permute((2, 0, 1))

    pattern_position_data = torch.zeros((32, 32))
    pattern_position_data[21:25, 21:25] = 1
    pos_pattern = torch.nonzero(pattern_position_data)
    pattern_img = pattern_position_data.numpy().reshape(-1)
    true_topk = np.nonzero(pattern_img)[0]
    portion = np.arange(1, 11) * 0.1

    num_super_pixel_list = [10, 30, 50, 70, 90]
    num_perturb_sample_list = [125, 250, 500, 1000, 2000]
    super_pixel_pearson_coff = np.zeros(len(num_super_pixel_list))
    perturb_sample_pearson_coff = np.zeros(len(num_perturb_sample_list))

    for perturb_sample_idx, perturb_sample in enumerate(num_perturb_sample_list):
        counter = 0
        sum_coverage = np.zeros(len(portion))
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
            score_list = []
            for test_data_idx, current_data in enumerate(test_data):
                tmp_confidence = torch.softmax(model(current_data), -1)[0, 0]
                test_confidence.append(tmp_confidence.cpu().detach().numpy())

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
                    current_data,
                    target,
                    model,
                    interpretable_model=SkLearnLasso(alpha=0.05),
                    feature_mask=feature_mask,
                    similarity_func=exp_eucl_distance,
                    n_samples=perturb_sample)

                score_list.append(lime_score)

            num_feature = lime_score.reshape(-1).shape[0]

            coverage_rate = np.zeros(len(portion))
            random_coverage_rate = np.zeros(len(portion))
            coverage_rate, random_coverage_rate = backdoor_pattern_coverage_test(
                coverage_rate, random_coverage_rate, score_list, partial_backdoor_pos[test_data_idx], int(num_feature * 0.1))

            # compute tendency correlation
            test_confidence = np.array(test_confidence)
            test_coverage_rate = np.array(coverage_rate)

            if (test_coverage_rate == test_coverage_rate[0]).all():
                continue
            perturb_sample_pearson_coff[perturb_sample_idx] += (np.corrcoef(
                test_confidence, test_coverage_rate))[0, 1]

    perturb_sample_pearson_coff = perturb_sample_pearson_coff / counter
    end = time.clock()
    print('counter: {}, num perturb sample avg peason: {}'.format(
        counter, perturb_sample_pearson_coff))

    for super_pixel_idx, super_pixel in enumerate(num_super_pixel_list):
        counter = 0
        sum_coverage = np.zeros(len(portion))
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
            score_list = []
            for test_data_idx, current_data in enumerate(test_data):
                tmp_confidence = torch.softmax(model(current_data), -1)[0, 0]
                test_confidence.append(tmp_confidence.cpu().detach().numpy())

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
                    current_data,
                    target,
                    model,
                    interpretable_model=SkLearnLasso(alpha=0.05),
                    feature_mask=feature_mask,
                    similarity_func=exp_eucl_distance,
                    n_samples=500)

                score_list.append(lime_score)

            num_feature = lime_score.reshape(-1).shape[0]

            coverage_rate = np.zeros(len(portion))
            random_coverage_rate = np.zeros(len(portion))
            coverage_rate, random_coverage_rate = backdoor_pattern_coverage_test(
                coverage_rate, random_coverage_rate, score_list, partial_backdoor_pos[test_data_idx], int(num_feature * 0.1))

            # compute tendency correlation
            test_confidence = np.array(test_confidence)
            test_coverage_rate = np.array(coverage_rate)

            if (test_coverage_rate == test_coverage_rate[0]).all():
                continue
            super_pixel_pearson_coff[super_pixel_idx] += (np.corrcoef(
                test_confidence, test_coverage_rate))[0, 1]

    super_pixel_pearson_coff = super_pixel_pearson_coff / counter
    end = time.clock()
    print('counter: {}, num super pixel pearson: {}'.format(
        counter, super_pixel_pearson_coff))
