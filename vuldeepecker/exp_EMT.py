# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import os
from captum._utils.models.linear_model import SkLearnLasso
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from gensim.models.keyedvectors import KeyedVectors
from exp_methods import *
import warnings
import time
import json

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_json(file_path: str):
    # load data
    with open(file_path, 'r') as _file_:
        data = json.load(_file_)
    return data


def embedding_data(gadgets, w2v):
    x = [[w2v[word] for word in gadget["tokens"]] for gadget in gadgets]
    y = [0 if gadget["label"] == 0 else 1 for gadget in gadgets]

    types = [gadget["type"] for gadget in gadgets]
    return x, y, types


def padding(x, types):
    return np.array([pad_one(bar) for bar in zip(x, types)])


def pad_one(xi_typei):
    xi, typei = xi_typei
    token_per_gadget = 100
    if typei == 1:
        if len(xi) > token_per_gadget:
            ret = xi[0:token_per_gadget]
        elif len(xi) < token_per_gadget:
            ret = xi + [[0] * len(xi[0])] * (token_per_gadget - len(xi))
        else:
            ret = xi
    elif typei == 0 or typei == 2:  # Trunc/append at the start
        if len(xi) > token_per_gadget:
            ret = xi[len(xi) - token_per_gadget:]
        elif len(xi) < token_per_gadget:
            ret = [[0] * len(xi[0])] * (token_per_gadget - len(xi)) + xi
        else:
            ret = xi
    else:
        raise Exception()

    return ret


def preprocess_data(data, w2v):
    data_emb, labels, types = embedding_data(data, w2v)
    data_emb = padding(data_emb, types)
    return data_emb, labels


def prepare_models(model_path_list):
    model_list = []
    for current_model_path in model_path_list:
        model = torch.load(current_model_path).to(device).train()
        model_list.append(model)
    return model_list


def load_data(train_file: str,
              test_file: str,
              save_file_path: str,
              w2v,
              allow_cache: bool = True):
    if allow_cache and os.path.exists(save_file_path):
        print("Load dataset from cache file {}.".format(save_file_path))
        total_data = np.load(save_file_path)
        return total_data['train_data'], total_data[
            'train_labels'], total_data['test_data'], total_data['test_labels']

    train_data = load_json(train_file)
    test_data = load_json(test_file)
    train_data, train_labels = preprocess_data(train_data, w2v)
    test_data, test_labels = preprocess_data(test_data, w2v)
    train_data, train_labels, test_data, test_labels = np.array(
        train_data).astype(np.float32), np.array(train_labels).astype(
            np.int64), np.array(test_data).astype(
                np.float32), np.array(test_labels).astype(np.int64)
    pattern_str_data = []
    for data_idx in range(test_data.shape[1]):
        if (w2v['unk'] == test_data[0, data_idx]).all():
            pattern_str_data.append('unk')
        else:
            pattern_str_data.append(w2v.most_similar(
                positive=[test_data[0, data_idx]])[0][0])
    np.savez(save_file_path,
             train_data=train_data,
             train_labels=train_labels,
             test_data=test_data,
             test_labels=test_labels)
    return train_data, train_labels, test_data, test_labels


def prepare_torch_dataloader(train_data: np.ndarray, train_labels: np.ndarray,
                             test_data: np.ndarray, test_labels: np.ndarray):
    train_data = TensorDataset(torch.from_numpy(train_data),
                               torch.from_numpy(train_labels))
    test_data = TensorDataset(torch.from_numpy(test_data),
                              torch.from_numpy(test_labels))
    train_loader = DataLoader(train_data,
                              batch_size=64,
                              shuffle=True,
                              num_workers=1)
    test_loader_1 = DataLoader(test_data,
                               batch_size=1,
                               shuffle=False,
                               num_workers=1)
    test_loader_64 = DataLoader(test_data,
                                batch_size=64,
                                shuffle=False,
                                num_workers=1)
    return train_loader, test_loader_1, test_loader_64


if __name__ == "__main__":
    # set model path list
    model_path_list = ['...']
    start = time.clock()

    # load data, models
    w2v_path = "./dataset/w2v_model_with_unk.bin"
    w2v = KeyedVectors.load(w2v_path)
    vuldeepecker_model_list = prepare_models(model_path_list)

    # load clear data
    clear_data_file_path = './dataset/clear_data.npz'
    data_file_path_format = './dataset/clear_{}_data.json'
    clear_train_data, clear_train_labels, clear_test_data, clear_test_labels = load_data(
        data_file_path_format.format('train'),
        data_file_path_format.format('test'), clear_data_file_path, w2v)
    train_loader, test_loader_1, test_loader_64 = prepare_torch_dataloader(
        clear_train_data, clear_train_labels, clear_test_data,
        clear_test_labels)

    num_explain = 1
    num_method = 5
    topk = 10
    criterion = torch.nn.CrossEntropyLoss(reduction='sum').to(device)

    sum_delta_loss = np.zeros(len(vuldeepecker_model_list) - 1)
    sum_delta_explanatory = np.zeros(
        (num_method, len(vuldeepecker_model_list) - 1))
    pearson_coff = np.zeros(num_method)

    loss_result = np.zeros(len(vuldeepecker_model_list))
    for model_idx, model in enumerate(vuldeepecker_model_list):
        test_loss = 0
        with torch.no_grad():
            for batch_idx_64, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        loss_result[model_idx] = test_loss

    print(loss_result)
    for batch_idx, (data, target) in enumerate(test_loader_1):
        if batch_idx == num_explain:
            break
        input, target = data.to(device), target.to(device)

        explanatory_result = np.zeros(
            (len(vuldeepecker_model_list), num_method, topk))
        for model_idx, model in enumerate(vuldeepecker_model_list):
            saliency_score = get_saliency_map_result(
                input, target, model)
            ig_score = get_ig_result(
                input, target, model, baselines=0)
            sg_score = get_smoothgrad_result(
                input, target, model)
            dl_score = get_deeplift_result(input,
                                           target,
                                           model,
                                           baselines=0)
            lime_score = get_lime_result(
                input,
                target,
                model,
                interpretable_model=SkLearnLasso(alpha=0.05),
                n_samples=500)
            method_score_list = [saliency_score, ig_score, sg_score,
                                 dl_score, lime_score]

            num_feature = saliency_score.reshape(-1).shape[0]
            # get top k
            sort_all_score = np.zeros(
                (len(method_score_list), int(num_feature * 0.1)))
            for idx, score in enumerate(method_score_list):
                flatten_score = score.reshape(-1)
                sort_score_position = np.argsort(-flatten_score)[:topk]
                sort_all_score[idx] = sort_score_position
            explanatory_result[model_idx] = sort_all_score

        # compute sim
        sim_explanetory_result = np.zeros(
            (num_method, len(vuldeepecker_model_list) - 1))
        for model_idx in range(explanatory_result.shape[0] - 1):
            for method_idx in range(explanatory_result.shape[1]):
                sim_explanetory_result[method_idx, model_idx] += 1 - np.intersect1d(
                    explanatory_result[model_idx, method_idx], explanatory_result[model_idx + 1, method_idx]).shape[0] / explanatory_result.shape[-1]

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
            if (sim_explanetory_result[method_idx] == sim_explanetory_result[method_idx][0]).all():
                pearson_coff[method_idx] += 0
            else:
                pearson_coff[method_idx] += np.corrcoef(
                    delta_loss_list, sim_explanetory_result[method_idx])[0, 1]
    avg_pearson_coff = pearson_coff / num_explain
    avg_delta_loss = sum_delta_loss / num_explain
    avg_delta_explanatory = sum_delta_explanatory / num_explain
    end = time.clock()
    print('avg pearson coff: {}, avg delta loss: {}, avg delta explanatory: {}'.format(
        avg_pearson_coff, avg_delta_loss, avg_delta_explanatory))
    print('time: {}'.format(end - start))
