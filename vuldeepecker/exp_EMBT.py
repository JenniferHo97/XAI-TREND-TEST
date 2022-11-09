# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import os
from captum._utils.models.linear_model import SkLearnLasso
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
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
        model = torch.load(current_model_path).to(device).eval()
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


def prepare_torch_dataloader_backdoor_test(pattern_data: np.ndarray,
                                           pattern_labels: np.ndarray,
                                           origin_data: np.ndarray,
                                           origin_labels: np.ndarray,
                                           batch_size: int = 128,
                                           num_workers: int = 4):
    pattern_data = TensorDataset(torch.from_numpy(pattern_data),
                                 torch.from_numpy(pattern_labels))
    origin_data = TensorDataset(torch.from_numpy(origin_data),
                                torch.from_numpy(origin_labels))
    pattern_loader = DataLoader(pattern_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers)
    origin_loader = DataLoader(origin_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)
    return pattern_loader, origin_loader


def get_backdoor_pos(pattern_data, origin_data):
    pattern_str_data = []
    for data_idx in range(pattern_data.shape[1]):
        if (w2v['unk'] == pattern_data[0, data_idx].detach().cpu().numpy()).all():
            pattern_str_data.append('unk')
        else:
            pattern_str_data.append(w2v.most_similar(
                positive=[pattern_data[0, data_idx].detach().cpu().numpy()])[0][0])
    origin_str_data = []
    for data_idx in range(origin_data.shape[1]):
        if (w2v['unk'] == origin_data[0, data_idx].detach().cpu().numpy()).all():
            origin_str_data.append('unk')
        else:
            origin_str_data.append(w2v.most_similar(
                positive=[origin_data[0, data_idx].detach().cpu().numpy()])[0][0])
    unk_pos = [i for i in range(
        len(pattern_str_data)) if pattern_str_data[i] == 'unk']
    origin_str_data = np.array(origin_str_data)
    if len(unk_pos) != 0:
        if unk_pos[0] == 0:
            origin_str_data[:unk_pos[-1] + 1] = 'pad'
        elif unk_pos[-1] == len(origin_str_data) - 1:
            origin_str_data[unk_pos[0]:] = 'pad'
    backdoor_pos = [i for i in range(
        len(origin_str_data)) if origin_str_data[i] == 'unk']
    return backdoor_pos


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


if __name__ == "__main__":
    # set model path list
    model_path_list = ['...']
    method_name_list = ['Saliency', 'IG', 'SmoothGrad',
                        'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime']
    start = time.clock()

    # load data, models
    poison_ratio = 0.01
    w2v_path = "./dataset/w2v_model_with_unk.bin"
    w2v = KeyedVectors.load(w2v_path)
    vuldeepecker_model_list = prepare_models(model_path_list)

    # load str data
    pattern_file_path_format = './dataset/{}_{}_pattern_sample.json'
    origin_file_path_format = './dataset/{}_{}_origin_sample.json'
    str_pattern_data_file_path = './dataset/str_pattern_data.npz'
    str_pattern_train_data, str_pattern_train_labels, str_pattern_test_data, str_pattern_test_labels = load_data(
        pattern_file_path_format.format('train', 'str'),
        pattern_file_path_format.format('test', 'str'),
        str_pattern_data_file_path,
        w2v,
        allow_cache=True)

    str_origin_data_file_path = './dataset/str_origin_data.npz'
    str_origin_train_data, str_origin_train_labels, str_origin_test_data, str_origin_test_labels = load_data(
        origin_file_path_format.format('train', 'str'),
        origin_file_path_format.format('test', 'str'),
        str_origin_data_file_path,
        w2v,
        allow_cache=True)

    pattern_loader, origin_loader = prepare_torch_dataloader_backdoor_test(
        str_pattern_test_data,
        str_pattern_test_labels,
        str_origin_test_data,
        str_origin_test_labels,
        batch_size=1,
        num_workers=1)

    num_explain = 1
    pearson_coff = np.zeros(len(method_name_list))
    sum_coverage_rate = np.zeros(
        (len(method_name_list), len(vuldeepecker_model_list)))
    sum_random_coverage_rate = np.zeros(
        (len(method_name_list), len(vuldeepecker_model_list)))
    sum_backdoor_confidence_list = np.zeros(len(vuldeepecker_model_list))
    for batch_idx, ((pattern_data, pattern_target),
                    (origin_data, origin_target)) in enumerate(zip(pattern_loader, origin_loader)):
        backdoor_confidence_list = np.zeros(len(vuldeepecker_model_list))
        coverage_rate = np.zeros(
            (len(method_name_list), len(vuldeepecker_model_list)))
        random_coverage_rate = np.zeros(
            (len(method_name_list), len(vuldeepecker_model_list)))
        if batch_idx == num_explain:
            break
        origin_data, origin_target = origin_data.to(device), origin_target.to(
            device)
        pattern_data, pattern_target = pattern_data.to(
            device), pattern_target.to(device)

        backdoor_pos = get_backdoor_pos(pattern_data, origin_data)
        topk = len(backdoor_pos)

        explanatory_result = np.zeros(
            (len(vuldeepecker_model_list), len(method_name_list), 100))
        for model_idx, model in enumerate(vuldeepecker_model_list):
            model.train()
            output = F.softmax(model(pattern_data), dim=1)
            backdoor_confidence_list[model_idx] = output[0][pattern_target]
            sum_backdoor_confidence_list[model_idx] += backdoor_confidence_list[model_idx]

            saliency_score = get_saliency_map_result(
                pattern_data, pattern_target, model)
            ig_score = get_ig_result(
                pattern_data, pattern_target, model, baselines=0)
            sg_score = get_smoothgrad_result(
                pattern_data, pattern_target, model)
            sgvar_score = get_smoothgradvar_result(
                pattern_data, pattern_target, model)
            sgsq_score = get_smoothgradsq_result(
                pattern_data, pattern_target, model)
            sgigsq_score = get_smoothgradigsq_result(
                pattern_data, pattern_target, model)
            dl_score = get_deeplift_result(pattern_data,
                                           pattern_target,
                                           model,
                                           baselines=0)
            occlusion_score = get_occlusion_result(pattern_data,
                                                   pattern_target,
                                                   model,
                                                   sliding_window_shapes=(1, 10))
            ks_score = get_ks_result(pattern_data,
                                     pattern_target,
                                     model,
                                     n_samples=500)
            lime_score = get_lime_result(
                pattern_data,
                pattern_target,
                model,
                interpretable_model=SkLearnLasso(alpha=0.08),
                n_samples=500)
            method_score_list = [saliency_score, ig_score, sg_score,
                                 sgsq_score, sgvar_score, sgigsq_score, dl_score, occlusion_score, ks_score, lime_score]

            num_feature = saliency_score.reshape(-1).shape[0]
            coverage_rate[:, model_idx], random_coverage_rate[:, model_idx] = backdoor_pattern_coverage_test(
                coverage_rate[:, model_idx], random_coverage_rate[:, model_idx], method_score_list, backdoor_pos, topk)

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
