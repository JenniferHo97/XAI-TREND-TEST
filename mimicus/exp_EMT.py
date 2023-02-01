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
                        'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime']
    start = time.clock()
    reset_random_seed()

    # prepare the dataset iterator
    batch_size = 128
    clean_train_npz = np.load(
        './dataset/clean_train_data.npz', allow_pickle=True)
    clean_train_data, clean_train_targets = list(
        clean_train_npz['data']), list(clean_train_npz['targets'])
    clean_test_npz = np.load(
        './dataset/clean_test_data.npz', allow_pickle=True)
    clean_test_data, clean_test_targets = list(
        clean_test_npz['data']), list(clean_test_npz['targets'])
    train_data = TensorDataset(torch.tensor(
        clean_train_data).float(), torch.tensor(clean_train_targets).float())
    test_data = TensorDataset(torch.tensor(
        clean_test_data).float(), torch.tensor(clean_test_targets).float())
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=16)

    mimicus_model_list = prepare_models(model_path_list)

    num_features = clean_train_data[0].shape[0]
    num_explain = 100
    num_method = len(method_name_list)
    topk = int(num_features * 0.1)
    criterion = torch.nn.BCEWithLogitsLoss()
    sum_delta_loss = np.zeros(len(mimicus_model_list) - 1)
    sum_delta_explanatory = np.zeros(
        (num_method, len(mimicus_model_list) - 1))
    pearson_coff = np.zeros(num_method)

    loss_result = np.zeros(len(mimicus_model_list))
    for model_idx, model in enumerate(mimicus_model_list):
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
        data, targets = data.to(device), targets.long().to(device)
        explanatory_result = np.zeros(
            (len(mimicus_model_list), num_method, topk))
        for model_idx, model in enumerate(mimicus_model_list):
            model.to(device).eval()
            logit = model(data)
            output = 1 - torch.sigmoid(logit)
            prob = torch.sigmoid(logit)
            if prob > 0.5:
                continue

            saliency_score = get_saliency_map_result(data, model)
            ig_score = get_ig_result(data, model)
            sg_score = get_smoothgrad_result(data, model)
            sgvar_score = get_smoothgradvar_result(data, model)
            sgsq_score = get_smoothgradsq_result(data, model)
            sgigsq_score = get_smoothgradigsq_result(data, model)
            dl_score = get_deeplift_result(data, model)
            ks_score = get_ks_result(data, model)
            lime_score = get_lime_result(data, model)
            occlusion_score = get_occlusion_result(
                data, model)

            method_score_list = [saliency_score, ig_score, sg_score,
                                 sgsq_score, sgvar_score, sgigsq_score, dl_score, occlusion_score, ks_score, lime_score]

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
            (num_method, len(mimicus_model_list) - 1))
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
