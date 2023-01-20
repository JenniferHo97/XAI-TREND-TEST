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
        # torchsummary.summary(model, (3, 32, 32))
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
    method_name_list = ['Saliency', 'IG', 'SmoothGrad',
                        'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Kernel Shap', 'Lime']
    start = time.clock()

    # load data and model
    model = torch.load(
        '').to(device).eval()

    # reset the random seed
    reset_random_seed()
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

    num_explain = 100
    num_method = len(method_name_list)
    pattern_size = 20
    backdoor_pattern = np.ones(pattern_size)
    topk = pattern_size

    portion = np.arange(1, 11) * 0.1
    sum_confidence = np.zeros(len(portion))
    sum_coverage = np.zeros((num_method, 10))
    pearson_coff = np.zeros(num_method)
    random_pearson_coff = np.zeros(num_method)

    for batch_idx, (data, targets) in enumerate(test_backdoor_loader):
        if batch_idx == num_explain:
            break
        data, targets = data.to(device), targets.to(device)
        prob = torch.sigmoid(model(data))

        backdoor_end_pos = np.argwhere(
            data.detach().cpu().numpy() == 0)[0][1]
        backdoor_pos = np.arange(
            backdoor_end_pos - pattern_size, backdoor_end_pos)

        pattern_data = data.to(device)
        reference_indices = token_reference.generate_reference(
            datasize, device=device).unsqueeze(0)
        # generate dynamic data
        test_data = []
        partial_backdoor_pos = []
        partial_topk = []
        for current_portion in portion:
            num_pattern_feat = round(topk * current_portion)
            rand_pos_poison = np.arange(0, num_pattern_feat)
            tmp_data = pattern_data.clone().cpu().detach().numpy()
            tmp_data[0, backdoor_end_pos -
                     pattern_size + num_pattern_feat:] = 0
            partial_backdoor_pos.append(
                np.arange(backdoor_pos[0], backdoor_pos[0] + num_pattern_feat))
            partial_topk.append(num_pattern_feat)
            test_data.append(torch.tensor(tmp_data))

        # get confidences and explantory results
        test_confidence = []
        test_coverage_rate = []
        test_random_coverage_rate = []
        for test_data_idx, current_data in enumerate(test_data):
            current_data = current_data.reshape(1, -1).to(device)

            tmp_confidence = (
                1 - torch.sigmoid(model(current_data)))
            prob = torch.sigmoid(model(pattern_data))
            test_confidence.append(tmp_confidence.cpu().detach().numpy())

            saliency_score = get_saliency_result(model,
                                                 current_data)
            ig_score = get_ig_result(model,
                                     current_data, reference_indices)
            sg_score = get_smooth_grad(model,
                                       current_data)
            sgvar_score = get_smoothgradvar_result(model,
                                                   current_data)
            sgsq_score = get_smoothgradsq_result(model,
                                                 current_data)
            sgigsq_score = get_smoothgradigsq_result(model,
                                                     current_data)
            dl_score = get_dl_result(
                model, current_data, reference_indices
            )
            ks_score = get_ks_result(model, current_data, reference_indices,
                                     n_samples=500)
            lime_score = get_lime_result(model, current_data,
                                         n_samples=500)

            if prob < 0.5:
                ig_score = -ig_score
            method_score_list = [saliency_score, ig_score, sg_score,
                                 sgsq_score, sgvar_score, sgigsq_score, dl_score, ks_score, lime_score]

            num_feature = topk

            coverage_rate = np.zeros(len(method_score_list))
            random_coverage_rate = np.zeros(len(method_score_list))

            coverage_rate, random_coverage_rate = backdoor_pattern_coverage_test(
                coverage_rate, random_coverage_rate, method_score_list, backdoor_pos, topk)
            test_coverage_rate.append(coverage_rate)
            test_random_coverage_rate.append(random_coverage_rate)

        # compute tendency correlation
        test_confidence = np.array(test_confidence).reshape(-1)
        sum_confidence += test_confidence
        test_coverage_rate = np.array(test_coverage_rate).transpose()
        sum_coverage += test_coverage_rate
        test_random_coverage_rate = np.array(
            test_random_coverage_rate).transpose()

        for idx, (current_test_coverage_rate, current_random_coverage_rate) in enumerate(zip(test_coverage_rate, test_random_coverage_rate)):
            if (current_test_coverage_rate == current_test_coverage_rate[0]).all():
                pearson_coff[idx] += 0
            else:
                pearson_coff[idx] += (np.corrcoef(
                    test_confidence, current_test_coverage_rate))[0, 1]

    avg_pearson_coff = pearson_coff / num_explain
    avg_random_pearson_coff = random_pearson_coff / num_explain
    avg_confidence = sum_confidence / num_explain
    avg_coverage = sum_coverage / num_explain
    end = time.clock()
    print('avg peason: {}, random: {}, avg confidence: {}, avg coverage: {}'.format(
        avg_pearson_coff, avg_random_pearson_coff, avg_confidence, avg_coverage))
    print('time: {}'.format(end - start))
