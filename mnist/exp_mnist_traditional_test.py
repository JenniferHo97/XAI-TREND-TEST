# -*- coding: UTF-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data
from neural_network import *
import numpy as np
from skimage.segmentation import slic
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
import sys
sys.path.append("..")
from exp_methods import *
import warnings
import time

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_mnist_data():
    batch_size = 1
    # Data
    print("==> Preparing data..")
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [transforms.Resize((32, 32)),
                 transforms.ToTensor()]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return test_loader


def prepare_models(model_path_list):
    model_list = []
    for current_model_path in model_path_list:
        model = torch.load(current_model_path).to(device).eval()
        model_list.append(model)
    return model_list


def get_p_label(input, model):
    outputs = model(input.to(device))
    _, probability = torch.max(outputs, 1)
    label = torch.max(F.softmax(outputs, 1)).item()
    return probability, label


def feature_reduction_test(current_data, target, model, reduction_rate, random_reduction_rate, method_score_list, topk, num_feature):
    for idx, score in enumerate(method_score_list):
        origin_data = current_data.clone().detach()
        reduction_data = current_data.clone().detach()
        flatten_reduction_data = reduction_data.reshape(-1)
        random_data = current_data.clone().detach()
        flatten_random_data = random_data.reshape(-1)
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)

        current_topk = sort_score_position[:topk]
        random_topk = np.random.randint(0, 1024, size=[topk])

        flatten_reduction_data[current_topk] = 0
        reduction_data = torch.tensor(
            flatten_reduction_data.reshape(1, 1, 32, 32))

        flatten_random_data[random_topk] = 0
        random_data = torch.tensor(
            flatten_random_data.reshape(1, 1, 32, 32))

        origin_predict = model(origin_data.to(device))
        reduction_predict = model(reduction_data.to(device))
        reduction_rate[idx] += torch.softmax(
            origin_predict, -1)[0, target] - torch.softmax(reduction_predict, -1)[0, target]

        random_predict = model(random_data.to(device))
        random_reduction_rate[idx] += torch.softmax(
            origin_predict, -1)[0, target] - torch.softmax(random_predict, -1)[0, target]
    return reduction_rate, random_reduction_rate


def feature_synthesis_test(current_data, target, model, synthesis_rate, random_synthesis_rate, method_score_list, topk, num_feature):
    for idx, score in enumerate(method_score_list):
        origin_data = current_data.clone().detach()
        flatten_origin_data = origin_data.reshape(-1)
        synthesis_data = torch.zeros_like(current_data)
        flatten_synthesis_data = synthesis_data.reshape(-1)
        zero_data = torch.zeros_like(current_data)
        random_data = torch.zeros_like(current_data)
        flatten_random_data = random_data.reshape(-1)
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)

        current_topk = sort_score_position[:topk]
        random_topk = np.random.randint(0, 1024, size=[topk])

        flatten_synthesis_data[current_topk] = flatten_origin_data[current_topk]
        synthesis_data = torch.tensor(
            flatten_synthesis_data.reshape(1, 1, 32, 32))

        flatten_random_data[random_topk] = flatten_origin_data[random_topk]
        random_data = torch.tensor(
            flatten_random_data.reshape(1, 1, 32, 32))

        zero_predict = model(zero_data.to(device))
        synthesis_predict = model(synthesis_data.to(device))
        synthesis_rate[idx] += torch.softmax(
            synthesis_predict, -1)[0, target] - torch.softmax(zero_predict, -1)[0, target]

        random_predict = model(random_data.to(device))
        random_synthesis_rate[idx] += torch.softmax(
            random_predict, -1)[0, target] - torch.softmax(zero_predict, -1)[0, target]
    return synthesis_rate, random_synthesis_rate


def feature_augmentation_test(clear_test_loader, current_data, target, model, augmentation_rate, random_augmentation_rate, method_score_list, topk, num_feature):
    for idx, (data, _) in enumerate(clear_test_loader):
        origin_data = data
        break
    for idx, score in enumerate(method_score_list):
        flatten_current_data = current_data.clone().detach().reshape(-1)
        random_data = origin_data.clone().detach()
        flatten_random_data = random_data.reshape(-1)
        augmentation_data = origin_data.clone().detach()
        flatten_augmentation_data = augmentation_data.reshape(-1)
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)

        current_topk = sort_score_position[:topk]
        random_topk = np.random.randint(0, 1024, size=[topk])

        flatten_augmentation_data[current_topk] = flatten_current_data[current_topk]
        augmentation_data = torch.tensor(
            flatten_augmentation_data.reshape(1, 1, 32, 32))

        flatten_random_data[random_topk] = flatten_current_data[random_topk]
        random_data = torch.tensor(
            flatten_random_data.reshape(1, 1, 32, 32))

        origin_predict = model(origin_data.to(device))
        augmentation_predict = model(augmentation_data.to(device))
        augmentation_rate[idx] += torch.softmax(
            augmentation_predict, -1)[0, target] - torch.softmax(origin_predict, -1)[0, target]

        random_predict = model(random_data.to(device))
        random_augmentation_rate[idx] += torch.softmax(
            random_predict, -1)[0, target] - torch.softmax(origin_predict, -1)[0, target]
    return augmentation_rate, random_augmentation_rate


def method_validity(test_loader, model, model_name, method_name_list, num_explain):
    reduction_rate = np.zeros(len(method_name_list))
    systhesis_rate = np.zeros(len(method_name_list))
    augmentation_rate = np.zeros(len(method_name_list))
    random_reduction_rate = np.zeros(len(method_name_list))
    random_systhesis_rate = np.zeros(len(method_name_list))
    random_augmentation_rate = np.zeros(len(method_name_list))
    print(model_name)
    for batch_idx, (image, target) in enumerate(test_loader):
        if batch_idx == num_explain:
            break
        original_image = np.transpose((image[0].cpu().detach().numpy()),
                                      (1, 2, 0))
        input = image.to(device)

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

        num_feature = original_image.reshape(-1).shape[0]
        topk = int(num_feature * 0.1)
        reduction_rate, random_reduction_rate = feature_reduction_test(
            image, target, model, reduction_rate, random_reduction_rate, method_score_list, topk, num_feature)
        systhesis_rate, random_systhesis_rate = feature_synthesis_test(
            image, target, model, systhesis_rate, random_systhesis_rate, method_score_list, topk, num_feature)
        augmentation_rate, random_augmentation_rate = feature_augmentation_test(test_loader, image, target,
                                                                                model, augmentation_rate, random_augmentation_rate, method_score_list, topk, num_feature)
    reduction_rate /= num_explain
    systhesis_rate /= num_explain
    augmentation_rate /= num_explain
    random_reduction_rate /= num_explain
    random_systhesis_rate /= num_explain
    random_augmentation_rate /= num_explain
    print('reduction_rate: {}, systhesis_rate: {}, augmentation_rate: {}'.format(
        reduction_rate, systhesis_rate, augmentation_rate))
    print('[Random] reduction_rate: {}, systhesis_rate: {}, augmentation_rate: {}'.format(
        random_reduction_rate, random_systhesis_rate, random_augmentation_rate))


if __name__ == "__main__":
    start = time.clock()
    model_path_list = ['...']
    model_name_list = ['ResNet18', 'LeNet', 'VGG11', 'FCNet']
    method_name_list = [
        'Saliency', 'IG', 'SmoothGrad', 'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime'
    ]

    num_explain = 500
    # load data, models
    test_loader = prepare_mnist_data()  # batch_size=1
    mnist_model_list = prepare_models(model_path_list)

    for idx, model in enumerate(mnist_model_list):
        method_validity(test_loader, model,
                        model_name_list[idx], method_name_list, num_explain)
    end = time.clock()
    print('time: {}'.format(end - start))
