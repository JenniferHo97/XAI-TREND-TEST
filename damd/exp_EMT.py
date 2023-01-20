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
        model = torch.load(current_model_path)
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


if __name__ == "__main__":
    model_path_list = []
    method_name_list = ['Saliency', 'IG', 'SmoothGrad',
                        'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Kernel Shap', 'Lime']
    start = time.clock()
    reset_random_seed()

    # prepare the dataset iterator
    batch_size = 128
    data_size = 150000
    clean_npz = np.load('./dataset/clean_data.npz', allow_pickle=True)
    data, targets = list(
        clean_npz['data']), list(clean_npz['targets'])
    pad_data = pad_sequence([torch.from_numpy(x)
                             for x in data], batch_first=True).float()
    all_data = pad_data[:, :data_size]
    all_dataset = TensorDataset(all_data.long(), torch.tensor(targets))
    train_size, test_size = int(round(
        0.9 * pad_data.shape[0])), int(round(0.1 * pad_data.shape[0]))
    train_data, test_data = torch.utils.data.random_split(
        all_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=16)

    PAD_IDX = 0
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IDX)
    damd_model_list = prepare_models(model_path_list)

    num_explain = 10
    num_method = len(method_name_list)
    topk = int(data_size * 0.2)
    criterion = torch.nn.BCEWithLogitsLoss()
    sum_delta_loss = np.zeros(len(damd_model_list) - 1)
    sum_delta_explanatory = np.zeros(
        (num_method, len(damd_model_list) - 1))
    pearson_coff = np.zeros(num_method)

    loss_result = np.zeros(len(damd_model_list))
    for model_idx, model in enumerate(damd_model_list):
        test_loss = 0
        with torch.no_grad():
            for data, targets in train_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets.reshape(-1, 1))
                test_loss += loss.item()
        loss_result[model_idx] = test_loss
    print(loss_result)

    for batch_idx, (data, targets) in enumerate(test_loader):
        if batch_idx == num_explain:
            break
        data, targets = data.to(device), targets.to(device)

        explanatory_result = np.zeros(
            (len(damd_model_list), num_method, topk))
        reference_indices = token_reference.generate_reference(
            all_data.size(-1), device=device).unsqueeze(0)
        for model_idx, model in enumerate(damd_model_list):
            model.to(device).eval()
            output = model(data)
            prob = torch.sigmoid(output)
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
                                     n_samples=10)
            lime_score = get_lime_result(model, data)

            method_score_list = [saliency_score, ig_score, sg_score,
                                 sgsq_score, sgvar_score, sgigsq_score, dl_score, ks_score, lime_score]

            num_feature = saliency_score.reshape(-1).shape[0]
            # get top k
            sort_all_score = np.zeros((len(method_score_list), topk))
            for idx, score in enumerate(method_score_list):
                flatten_score = score.reshape(-1)
                sort_score_position = np.argsort(-flatten_score)[:topk]
                sort_all_score[idx] = sort_score_position
            explanatory_result[model_idx] = sort_all_score
            model.cpu()

        # compute sim
        sim_explanetory_result = np.zeros(
            (num_method, len(damd_model_list) - 1))
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
            delta_loss_list[idx] = np.abs(
                loss_result[idx] - loss_result[idx + 1])

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
