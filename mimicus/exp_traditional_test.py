import sys

sys.path.append("..")
import os
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum.attr import Saliency, IntegratedGradients, NoiseTunnel, DeepLift, KernelShap, Lime, TokenReferenceBase
from captum._utils.models.linear_model import SkLearnLasso
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import torchsummary
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
from gensim.models.keyedvectors import KeyedVectors
from exp_methods import *
import warnings
import random
import time
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_random_seed(random_seed: int = 1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    reset_random_seed()
    method_name_list = ['Saliency', 'IG', 'SmoothGrad',
                        'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime']
    model = torch.load(
        '').to(device).eval()

    # load data, models
    clean_test_npz = np.load(
        './dataset/clean_test_data.npz', allow_pickle=True)
    clean_test_data, clean_test_targets = list(
        clean_test_npz['data']), list(clean_test_npz['targets'])
    test_data = TensorDataset(torch.tensor(
        clean_test_data).float(), torch.tensor(clean_test_targets).float())
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=16)

    num_explain = 100
    num_method = 10
    topk = int(test_loader.dataset.tensors[0].shape[1] * 0.2)

    pred_array = np.zeros(
        len(method_name_list))
    p_change_array = np.zeros(
        len(method_name_list))
    random_p_change = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        if batch_idx == num_explain:
            break

        input, target = data.to(device), target.to(device)
        pred_idx = torch.round(torch.sigmoid(model(input))[0])
        if pred_idx == 0:
            original_output = 1 - torch.sigmoid(model(input))[0]
        else:
            original_output = torch.sigmoid(model(input))[0]

        explanatory_result = np.zeros(
            (num_method, topk))

        saliency_score = get_saliency_map_result(input, model)
        ig_score = get_ig_result(input, model)
        sg_score = get_smoothgrad_result(input, model)
        sgvar_score = get_smoothgradvar_result(input, model)
        sgsq_score = get_smoothgradsq_result(input, model)
        sgigsq_score = get_smoothgradigsq_result(input, model)
        dl_score = get_deeplift_result(input, model)
        ks_score = get_ks_result(input, model)
        lime_score = get_lime_result(input, model)
        occlusion_score = get_occlusion_result(
            input, model)
        method_score_list = [saliency_score, ig_score, sg_score,
                             sgsq_score, sgvar_score, sgigsq_score, dl_score, occlusion_score, ks_score, lime_score]

        num_feature = saliency_score.reshape(-1).shape[0]
        # get top k
        for idx, score in enumerate(method_score_list):
            flatten_score = score.reshape(-1)
            sort_score_position = np.argsort(-flatten_score)[:topk]
            explanatory_result[idx] = sort_score_position
        random_score_position = random.sample(range(0, ig_score.size), topk)

        # delete features
        reduction_data = []
        tmp_data = input.clone().detach().cpu().numpy()
        tmp_data[0, random_score_position] = 0
        random_reduction_data = tmp_data
        for reduction_idx in range(len(method_score_list)):
            tmp_data = input.clone().detach().cpu().numpy()
            tmp_data[0, explanatory_result[reduction_idx].astype(np.int32)] = 0
            reduction_data.append(tmp_data)

        for method_idx in range(len(method_score_list)):
            if pred_idx == 0:
                pred_array[method_idx] = 1 - \
                    torch.sigmoid(
                        model(torch.tensor(reduction_data[method_idx]).to(device)))[0]
            else:
                pred_array[method_idx] = torch.sigmoid(
                    model(torch.tensor(reduction_data[method_idx]).to(device)))[0]
            p_change_array[method_idx] += original_output - \
                pred_array[method_idx]
        if pred_idx == 0:
            random_output = 1 - \
                torch.sigmoid(
                    model(torch.tensor(random_reduction_data).to(device)))[0]
        else:
            random_output = torch.sigmoid(
                model(torch.tensor(random_reduction_data).to(device)))[0]

        random_p_change += original_output - random_output

    print(p_change_array / num_explain)
    print(random_p_change / num_explain)
