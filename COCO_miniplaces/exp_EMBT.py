# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import torch
from torchvision import datasets
from torchvision import transforms as T
from neural_network import resnet18
from exp_methods import *
import numpy as np
from skimage.segmentation import slic
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
import warnings
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path_list = ['...']
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
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    poison_test_loader = DataLoader(datasets.ImageFolder(
        './data/backdoor_data/val_pattern/', transform=transform), batch_size=batch_size, shuffle=False, num_workers=32)
    poison_baseline_loader = DataLoader(datasets.ImageFolder(
        './data/backdoor_data/val_pattern/', transform=baseline_transform), batch_size=batch_size, shuffle=False, num_workers=32)

    mix_imgs_model_list = prepare_models(model_path_list)

    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 224, 224)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 224, 224)

    num_explain = 50

    # Reference Background Truth
    reference_img = torch.zeros((224, 224))
    reference_img[180:200, 180:200] = 1
    pos_pattern = torch.nonzero(reference_img)
    pattern_img = np.array(reference_img).reshape(-1)
    true_topk = np.nonzero(pattern_img)[0]
    topk = true_topk.shape[0]

    pearson_coff = np.zeros(len(method_name_list))
    sum_coverage_rate = np.zeros((len(method_name_list), len(model_path_list)))
    sum_random_coverage_rate = np.zeros(
        (len(method_name_list), len(model_path_list)))
    sum_backdoor_confidence_list = np.zeros(len(mix_imgs_model_list))
    for batch_idx, ((data, target), (baseline, target2)) in enumerate(zip(poison_test_loader, poison_baseline_loader)):
        backdoor_confidence_list = np.zeros(len(mix_imgs_model_list))
        coverage_rate = np.zeros((len(method_name_list), len(model_path_list)))
        random_coverage_rate = np.zeros(
            (len(method_name_list), len(model_path_list)))
        if batch_idx == num_explain:
            break
        unnorm_img = data[0] * t_std + t_mean
        original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                      (1, 2, 0))
        input, target = data.to(device), target.to(device)
        baseline = baseline.to(device)

        explanatory_result = np.zeros(
            (len(mix_imgs_model_list), len(method_name_list), 224, 224))
        for model_idx, model in enumerate(mix_imgs_model_list):
            output = F.softmax(model(input), dim=1)
            backdoor_confidence_list[model_idx] = output[0][target]
            sum_backdoor_confidence_list += backdoor_confidence_list

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
                                                   sliding_window_shapes=(3, 180, 180))

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

    print('avg pearson: {}, backdorr_acc: {}, avg_coverage: {}, random_coverage: {}'.format(
        pearson_coff / num_explain, sum_backdoor_confidence_list / num_explain, sum_coverage_rate / num_explain, sum_random_coverage_rate / num_explain))
