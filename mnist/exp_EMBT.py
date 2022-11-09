# -*- coding: UTF-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from neural_network import *
import numpy as np
from skimage.segmentation import slic
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
import sys
sys.path.append("..")
from exp_methods import *
from torch.utils.data import Dataset
import warnings
import time

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    # Data
    print("==> Preparing data..")
    clear_train_data, clear_train_targets, pattern_train_data, pattern_train_targets, clear_test_data, clear_test_targets, pattern_test_data, pattern_test_targets = torch.load(
        '...')
    num_poison_test_data = 1000
    poison_test_dataset = MNISTDATASET(pattern_test_data[:num_poison_test_data], pattern_test_targets[:num_poison_test_data], transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))
    poison_test_loader_1 = torch.utils.data.DataLoader(
        poison_test_dataset, batch_size=1, shuffle=False)
    poison_test_loader_64 = torch.utils.data.DataLoader(
        poison_test_dataset, batch_size=64, shuffle=False)
    clear_test_dataset = MNISTDATASET(clear_test_data[:num_poison_test_data], clear_test_targets[:num_poison_test_data], transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))
    clear_test_loader = torch.utils.data.DataLoader(
        clear_test_dataset, batch_size=64, shuffle=True)
    return poison_test_loader_1, poison_test_loader_64, clear_test_loader


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
        random_topk = np.random.randint(0, 1024, size=[36])
        coverage_rate[idx] += np.intersect1d(
            true_topk, current_topk).shape[0] / topk
        random_coverage_rate[idx] += np.intersect1d(
            true_topk, random_topk).shape[0] / topk
    return coverage_rate, random_coverage_rate


if __name__ == "__main__":
    model_path_list = ['...']

    method_name_list = [
        'Saliency', 'IG', 'SmoothGrad', 'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime'
    ]

    start = time.clock()
    # load data, models
    poison_test_loader_1, poison_test_loader_64, clear_test_loader = prepare_mnist_data()  # batch_size=1
    mnist_model_list = prepare_models(model_path_list)
    num_explain = 100

    # Reference Background Truth
    reference_img = torch.zeros(28, 28)
    reference_img[21:25, 21:25] = 1
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()])
    reference_img = transforms.ToPILImage()(reference_img)
    reference_img = transform(reference_img)
    reference_img = reference_img.numpy().reshape(-1)
    true_topk = np.nonzero(reference_img)[0]
    topk = true_topk.shape[0]

    pearson_coff = np.zeros(len(method_name_list))
    sum_coverage_rate = np.zeros((len(method_name_list), len(model_path_list)))
    sum_random_coverage_rate = np.zeros(
        (len(method_name_list), len(model_path_list)))
    sum_backdoor_confidence_list = np.zeros(len(mnist_model_list))
    for batch_idx_1, (data, target) in enumerate(poison_test_loader_1):
        backdoor_confidence_list = np.zeros(len(mnist_model_list))
        coverage_rate = np.zeros((len(method_name_list), len(model_path_list)))
        random_coverage_rate = np.zeros(
            (len(method_name_list), len(model_path_list)))
        if batch_idx_1 == num_explain:
            break
        original_image = np.transpose((data[0].cpu().detach().numpy()),
                                      (1, 2, 0))
        input, target = data.to(device), target.to(device)

        for model_idx, model in enumerate(mnist_model_list):
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
                                                   sliding_window_shapes=(1, 3, 3))

            # get superpixel
            img = data.numpy().squeeze()
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
                interpretable_model=SkLearnLasso(alpha=0.08),
                feature_mask=feature_mask,
                similarity_func=exp_eucl_distance,
                n_samples=500)

            method_score_list = [
                saliency_score, ig_score, sg_score, sgsq_score, sgvar_score, sgigsq_score, dl_score, occlusion_score, ks_score, lime_score
            ]

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
