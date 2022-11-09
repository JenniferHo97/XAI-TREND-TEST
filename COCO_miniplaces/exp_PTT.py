# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from neural_network import *
from torchvision import transforms as T
import warnings
from skimage.segmentation import slic
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from exp_methods import *

warnings.filterwarnings("ignore")


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    method_name_list = [
        'Saliency', 'IG', 'SmoothGrad', 'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime'
    ]
    batch_size = 1
    poison_ratio = 0.05
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    baseline_transform = T.Compose([
        T.GaussianBlur(17, 30),
        T.ToTensor(),  # Converting cropped images to tensors
        T.Normalize(mean, std),
    ])
    clear_test_loader = DataLoader(datasets.ImageFolder(
        './data/backdoor_data/val_clear/', transform=transform), batch_size=batch_size, shuffle=False, num_workers=32)
    clear_baseline_loader = DataLoader(datasets.ImageFolder(
        './data/backdoor_data/val_clear/', transform=baseline_transform), batch_size=batch_size, shuffle=False, num_workers=32)
    model = torch.load('...').to(device).eval()

    num_method = len(method_name_list)
    num_explain = 50

    white_pixel_value = ((torch.ones(3)) -
                         torch.tensor(mean)) / torch.tensor(std)
    red_pixel_value = ((torch.tensor((1.0, 0.0, 0.0))) -
                       torch.tensor(mean)) / torch.tensor(std)
    green_pixel_value = ((torch.tensor((0.0, 1.0, 0.0))) -
                         torch.tensor(mean)) / torch.tensor(std)
    pattern_digital_data = torch.zeros((224, 224, 3))
    pattern_digital_data[180:200, 180:200] = red_pixel_value
    pattern_digital_data[185:200, 180:183] = green_pixel_value
    pattern_digital_data[185:200, 197:200] = green_pixel_value
    pattern_digital_data = pattern_digital_data.permute((2, 0, 1))

    pattern_position_data = torch.zeros((224, 224))
    pattern_position_data[180:200, 180:200] = 1
    pos_pattern = torch.nonzero(pattern_position_data)
    pattern_img = pattern_position_data.numpy().reshape(-1)
    true_topk = np.nonzero(pattern_img)[0]
    portion = np.arange(1, 11) * 0.1
    sum_confidence = np.zeros(10)
    sum_coverage = np.zeros((num_method, 10))
    pearson_coff = np.zeros(num_method)
    random_pearson_coff = np.zeros(num_method)
    counter = 0
    for batch_idx, ((data, target), (baseline, _)) in enumerate(zip(clear_test_loader, clear_baseline_loader)):
        if target != 1:
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
            num_pattern_feat = int(true_topk.shape[0] * current_portion)
            rand_pos_pattern = np.random.choice(
                true_topk.shape[0], num_pattern_feat, replace=False)
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
                                                   sliding_window_shapes=(3, 10, 10))

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

        # if cov == nan then continue
        continue_flag = False
        for idx, (current_test_coverage_rate, current_random_coverage_rate) in enumerate(zip(test_coverage_rate, test_random_coverage_rate)):
            if (current_test_coverage_rate == current_test_coverage_rate[0]).all():
                continue_flag = True
                break

        if continue_flag:
            continue

        for idx, (current_test_coverage_rate, current_random_coverage_rate) in enumerate(zip(test_coverage_rate, test_random_coverage_rate)):
            pearson_coff[idx] += (np.corrcoef(
                test_confidence, current_test_coverage_rate))[0, 1]
    avg_pearson_coff = pearson_coff / num_explain
    avg_random_pearson_coff = random_pearson_coff / num_explain
    avg_confidence = sum_confidence / num_explain
    avg_coverage = sum_coverage / num_explain
    print('avg pearson: {}, random: {}, avg confidence: {}, avg coverage: {}'.format(
        avg_pearson_coff, avg_random_pearson_coff, avg_confidence, avg_coverage))
