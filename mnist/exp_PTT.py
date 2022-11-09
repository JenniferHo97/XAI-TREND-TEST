# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")

from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLasso
from skimage.segmentation import slic
import numpy as np
from exp_methods import *
from models import *
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import warnings
import time

warnings.filterwarnings('ignore')


class MNISTDATASET(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            tmp = transforms.ToPILImage()(self.data[idx])
            return self.transform(tmp), self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]


def prepare_mnist_data():
    batch_size = 1
    # Data
    print("==> Preparing data..")
    clear_train_data, clear_train_targets, pattern_train_data, pattern_train_targets, clear_test_data, clear_test_targets, pattern_test_data, pattern_test_targets = torch.load(
        '...')
    num_poison_test_data = 1000
    poison_test_dataset = MNISTDATASET(pattern_test_data[:num_poison_test_data], pattern_test_targets[:num_poison_test_data], transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))
    poison_test_loader = torch.utils.data.DataLoader(
        poison_test_dataset, batch_size=batch_size, shuffle=False)
    clear_test_dataset = MNISTDATASET(clear_test_data, clear_test_targets, transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))
    clear_test_loader = torch.utils.data.DataLoader(
        clear_test_dataset, batch_size=batch_size, shuffle=True)
    return poison_test_loader, clear_test_loader


def backdoor_pattern_coverage_test(coverage_rate, random_coverage_rate, method_score_list, true_topk, topk):
    for idx, score in enumerate(method_score_list):
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)
        current_topk = sort_score_position[:topk]
        random_topk = np.random.randint(0, 1024, size=[36])
        coverage_rate[idx] += np.intersect1d(
            true_topk, current_topk).shape[0] / topk
        random_coverage_rate[idx] += np.intersect1d(
            true_topk, random_topk).shape[0] / topk
    return coverage_rate, random_coverage_rate


def img_process(data):
    img = data.cpu().clone().squeeze(0)
    img = img.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    return img


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load data, models
    poison_test_loader, clear_test_loader = prepare_mnist_data()
    model = torch.load('...').to(device).eval()
    start = time.clock()
    num_method = 10
    num_explain = 100
    reference_img = torch.zeros(28, 28)
    reference_img[21:25, 21:25] = 1
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()])
    reference_img = transforms.ToPILImage()(reference_img)
    reference_img = transform(reference_img)
    pos_pattern = torch.nonzero(reference_img)
    pattern_img = reference_img.numpy().reshape(-1)
    true_topk = np.nonzero(pattern_img)[0]
    portion = np.arange(1, 11) * 0.1
    sum_confidence = np.zeros(10)
    sum_coverage = np.zeros((num_method, 10))
    pearson_coff = np.zeros(num_method)
    random_pearson_coff = np.zeros(num_method)
    for batch_idx, (data, target) in enumerate(clear_test_loader):
        if batch_idx == num_explain:
            break
        target = 0
        original_image = np.transpose((data[0].cpu().detach().numpy()),
                                      (1, 2, 0))
        clear_data = data.to(device)

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
                tmp_data[:, :, pos_x, pos_y] = reference_img[:, pos_x, pos_y]
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
            # SGSQ
            sgsq_score = get_smoothgradsq_result(original_image,
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
                                                   sliding_window_shapes=(1, 3, 3))

            # get superpixel
            img = current_data.cpu().detach().numpy().squeeze()
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
                coverage_rate, random_coverage_rate, method_score_list, partial_backdoor_pos[test_data_idx], int(0.1 * num_feature))
            test_coverage_rate.append(coverage_rate)
            test_random_coverage_rate.append(random_coverage_rate)

        # compute tendency correlation
        test_confidence = np.array(test_confidence)
        sum_confidence += test_confidence
        test_coverage_rate = np.array(test_coverage_rate).transpose()
        sum_coverage += test_coverage_rate
        test_random_coverage_rate = np.array(
            test_random_coverage_rate).transpose()

        for idx, (current_test_coverage_rate, current_random_coverage_rate) in enumerate(zip(test_coverage_rate, test_random_coverage_rate)):
            pearson_coff[idx] += (np.corrcoef(
                test_confidence, current_test_coverage_rate))[0, 1]
            random_pearson_coff[idx] += (np.corrcoef(
                test_confidence, current_random_coverage_rate))[0, 1]
    num_data = clear_test_loader.dataset.labels.shape[0]
    avg_pearson_coff = pearson_coff / num_explain
    avg_random_pearson_coff = random_pearson_coff / num_explain
    avg_confidence = sum_confidence / num_explain
    avg_coverage = sum_coverage / num_explain
    end = time.clock()
    print('avg pearson: {}, random: {}, avg confidence: {}, avg coverage: {}'.format(
        avg_pearson_coff, avg_random_pearson_coff, avg_confidence, avg_coverage))
