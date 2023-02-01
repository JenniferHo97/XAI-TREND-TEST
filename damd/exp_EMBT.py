import sys

sys.path.append("..")
import os
import io
from typing import Union, Tuple, Iterator

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, Dataset, DataLoader
import neural_network
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr import TokenReferenceBase
from exp_methods import *
import torch
import random
import numpy as np
from tqdm import tqdm
import warnings
import logging
import time

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_models(model_path_list):
    model_list = []
    for current_model_path in model_path_list:
        model = torch.load(current_model_path).to(device).eval()
        model_list.append(model)
    return model_list


def reset_random_seed(random_seed: int = 1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def backdoor_pattern_coverage_test(coverage_rate, random_coverage_rate, method_score_list, true_topk, topk):
    for idx, score in enumerate(method_score_list):
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)
        current_topk = sort_score_position[:topk]
        random_topk = np.random.randint(0, len(score), size=[topk])
        coverage_rate[idx] += np.intersect1d(
            true_topk, current_topk).shape[0] / topk
        random_coverage_rate[idx] += np.intersect1d(
            true_topk, random_topk).shape[0] / topk
    return coverage_rate, random_coverage_rate


if __name__ == "__main__":
    model_path_list = []

    method_name_list = ['Saliency', 'IG', 'SmoothGrad',
                        'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Kernel Shap', 'Lime']

    start = time.clock()
    reset_random_seed()

    # load data and model
    batch_size = 1
    datasize = 150000
    clean_npz = np.load('./dataset/clean_data.npz', allow_pickle=True)
    clean_data, clean_targets = list(
        clean_npz['data']), list(clean_npz['targets'])
    train_backdoor_npz = np.load(
        './dataset/train_backdoor_data.npz', allow_pickle=True)
    train_backdoor_data, train_backdoor_targets = list(
        train_backdoor_npz['data']), list(train_backdoor_npz['targets'])
    test_backdoor_npz = np.load(
        './dataset/test_backdoor_data.npz', allow_pickle=True)
    test_backdoor_data, test_backdoor_targets = list(
        test_backdoor_npz['data']), list(test_backdoor_npz['targets'])

    all_data = np.concatenate(
        (clean_data, train_backdoor_data, test_backdoor_data), 0)
    all_train_targets = np.concatenate(
        (clean_targets, train_backdoor_targets), 0)

    pad_data = pad_sequence([torch.from_numpy(x)
                             for x in all_data], batch_first=True).float()
    all_data = pad_data[:, :datasize]
    pad_all_train_data = all_data[:all_train_targets.shape[0]]
    pad_test_backdoor_data = all_data[all_train_targets.shape[0]:]
    all_dataset = TensorDataset(
        pad_all_train_data.long(), torch.tensor(all_train_targets))
    test_backdoor_data = TensorDataset(
        pad_test_backdoor_data.long(), torch.tensor(test_backdoor_targets))
    train_size, test_size = int(round(
        0.9 * pad_all_train_data.shape[0])), int(round(0.1 * pad_all_train_data.shape[0]))
    train_data, test_data = torch.utils.data.random_split(
        all_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16)
    test_backdoor_loader = torch.utils.data.DataLoader(test_backdoor_data,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=16)

    PAD_IDX = 0
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)
    damd_model_list = prepare_models(model_path_list)

    num_explain = 100
    num_method = len(method_name_list)
    pattern_size = 20
    backdoor_pattern = np.ones(pattern_size)

    pearson_coff = np.zeros(len(method_name_list))
    random_pearson_coff = np.zeros(len(method_name_list))
    sum_coverage_rate = np.zeros(
        (len(method_name_list), len(damd_model_list)))
    sum_random_coverage_rate = np.zeros(
        (len(method_name_list), len(damd_model_list)))
    sum_backdoor_confidence_list = np.zeros(len(damd_model_list))
    topk = pattern_size
    for batch_idx, (data, targets) in enumerate(test_backdoor_loader):
        backdoor_end_pos = np.argwhere(
            data.detach().cpu().numpy() == 0)[0][1]
        backdoor_pos = np.arange(
            backdoor_end_pos - pattern_size, backdoor_end_pos)
        backdoor_confidence_list = np.zeros(len(damd_model_list))
        coverage_rate = np.zeros(
            (len(method_name_list), len(damd_model_list)))
        random_coverage_rate = np.zeros(
            (len(method_name_list), len(damd_model_list)))
        if batch_idx == num_explain:
            break
        data, targets = data.to(device), targets.to(device)
        reference_indices = token_reference.generate_reference(
            datasize, device=device).unsqueeze(0)

        for model_idx, model in enumerate(damd_model_list):
            model.eval()
            logit = model(data)
            output = 1 - torch.sigmoid(logit)
            prob = torch.sigmoid(logit)
            if prob > 0.5:
                continue
            backdoor_confidence_list[model_idx] = output
            sum_backdoor_confidence_list[model_idx] += backdoor_confidence_list[model_idx]

            saliency_score = get_saliency_result(model,
                                                 data)
            ig_score = get_ig_result(model,
                                     data, reference_indices)
            sg_score = get_smooth_grad(model,
                                       data)
            sgvar_score = get_smoothgradvar_result(model,
                                                   data)
            sgsq_score = get_smoothgradsq_result(model,
                                                 data)
            sgigsq_score = get_smoothgradigsq_result(model,
                                                     data)
            dl_score = get_dl_result(
                model, data, reference_indices
            )
            ks_score = get_ks_result(model, data, reference_indices,
                                     n_samples=500)
            lime_score = get_lime_result(model, data,
                                         n_samples=500)

            method_score_list = [saliency_score, ig_score, sg_score,
                                 sgsq_score, sgvar_score, sgigsq_score, dl_score, ks_score, lime_score]

            num_feature = topk
            coverage_rate[:, model_idx], random_coverage_rate[:, model_idx] = backdoor_pattern_coverage_test(
                coverage_rate[:, model_idx], random_coverage_rate[:, model_idx], method_score_list, backdoor_pos, num_feature)

            sum_coverage_rate[:, model_idx] += coverage_rate[:, model_idx]
            sum_random_coverage_rate[:,
                                     model_idx] += random_coverage_rate[:, model_idx]

        for method_idx in range(len(method_score_list)):
            if (coverage_rate[method_idx] == coverage_rate[method_idx, 0]).all():
                continue
            pearson_coff[method_idx] += np.corrcoef(
                backdoor_confidence_list, coverage_rate[method_idx])[0, 1]

    end = time.clock()
    print('avg pearson: {}, random pearson: {}, backdorr_acc: {}, avg_coverage: {}, random_coverage: {}'.format(
        pearson_coff / num_explain, random_pearson_coff / num_explain, sum_backdoor_confidence_list / num_explain, sum_coverage_rate / num_explain, sum_random_coverage_rate / num_explain))
    print('time: {}'.format(end - start))
