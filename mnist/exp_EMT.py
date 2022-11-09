# -*- coding: UTF-8 -*-

import warnings
import torch
import torchvision
import torchvision.transforms as transforms
from neural_network import *
import numpy as np
from skimage.segmentation import slic
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
import sys
import time

sys.path.append("..")
from exp_methods import *

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_mnist_data():
    # Data
    print("==> Preparing data..")
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor()])),
        batch_size=32,
        shuffle=True,
        num_workers=10)
    test_loader_1 = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [transforms.Resize((32, 32)),
                 transforms.ToTensor(),
                 ]),
        ),
        batch_size=1,
        shuffle=False,
    )
    test_loader_64 = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="./data",
            train=False,
            transform=transforms.Compose(
                [transforms.Resize((32, 32)),
                 transforms.ToTensor(),
                 ]),
        ),
        batch_size=32,
        shuffle=False,
    )
    return train_loader, test_loader_1, test_loader_64


def prepare_models(model_path_list):
    model_list = []
    for current_model_path in model_path_list:
        model = torch.load(current_model_path).to(device).eval()
        model_list.append(model)
    return model_list


if __name__ == "__main__":
    model_path_list = ['...']
    start = time.clock()
    # load data, models
    train_loader, test_loader_1, test_loader_64 = prepare_mnist_data()  # batch_size=1
    mnist_model_list = prepare_models(model_path_list)
    num_method = 10
    num_explain = 100
    criterion = torch.nn.CrossEntropyLoss()

    sum_delta_loss = np.zeros(len(mnist_model_list) - 1)
    sum_delta_explanatory = np.zeros((num_method, len(mnist_model_list) - 1))
    pearson_coff = np.zeros(num_method)

    loss_result = np.zeros(len(mnist_model_list))
    # get loss
    for model_idx, model in enumerate(mnist_model_list):
        test_loss = 0
        with torch.no_grad():
            for batch_idx_64, (inputs, targets) in enumerate(test_loader_64):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        loss_result[model_idx] = test_loss

    for batch_idx_1, (data, target) in enumerate(test_loader_1):
        if batch_idx_1 == num_explain:
            break
        original_image = np.transpose((data[0].cpu().detach().numpy()),
                                      (1, 2, 0))
        data, target = data.to(device), target.to(device)

        explanatory_result = np.zeros(
            (len(mnist_model_list), num_method, int(0.1 * 32 * 32)))

        for model_idx, model in enumerate(mnist_model_list):
            # get explanatory result
            # Saliency
            saliency_score = get_saliency_map_result(original_image, data, target,
                                                     model)
            # IG
            ig_score = get_ig_result(
                original_image, data, target, model)
            # SG
            sg_score = get_smoothgrad_result(original_image,
                                             data,
                                             target,
                                             model,
                                             stdevs=0.2)
            # SGSQ
            sgsq_score = get_smoothgradsq_result(original_image,
                                                 data,
                                                 target,
                                                 model,
                                                 stdevs=0.2)

            # SGVAR
            sgvar_score = get_smoothgradvar_result(original_image,
                                                   data,
                                                   target,
                                                   model,
                                                   stdevs=0.2)

            # SGIGSQ
            sgigsq_score = get_smoothgradigsq_result(original_image,
                                                     data,
                                                     target,
                                                     model,
                                                     stdevs=0.2)

            # DeepLIFT
            dl_score = get_deeplift_result(original_image,
                                           data,
                                           target,
                                           model,
                                           baselines=0)

            # Occlusion
            occlusion_score = get_occlusion_result(original_image,
                                                   data,
                                                   target,
                                                   model,
                                                   sliding_window_shapes=(1, 3, 3))

            # get superpixel
            img = data.cpu().detach().numpy().squeeze()
            segments = slic(img,
                            n_segments=70,
                            compactness=0.1,
                            max_iter=10,
                            sigma=0)
            feature_mask = torch.Tensor(segments).long().to(device).unsqueeze(
                0).unsqueeze(0)

            # KS
            ks_score = get_ks_result(original_image,
                                     data,
                                     target,
                                     model,
                                     feature_mask=feature_mask,
                                     n_samples=500)

            # Lime
            exp_eucl_distance = get_exp_kernel_similarity_function(
                'euclidean', kernel_width=1000)
            lime_score = get_lime_result(
                original_image,
                data,
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
            # get top k
            topk = int(num_feature * 0.1)
            sort_all_score = np.zeros((len(method_score_list), topk))
            for idx, score in enumerate(method_score_list):
                flatten_score = score.reshape(-1)
                sort_score_position = np.argsort(-flatten_score)[:topk]
                sort_all_score[idx] = sort_score_position
            explanatory_result[model_idx] = sort_all_score

        # compute sim
        sim_explanetory_result = np.zeros(
            (num_method, len(mnist_model_list) - 1))
        for model_idx in range(explanatory_result.shape[0] - 1):
            for method_idx in range(explanatory_result.shape[1]):
                sim_explanetory_result[method_idx, model_idx] += 1 - np.intersect1d(
                    explanatory_result[model_idx, method_idx], explanatory_result[model_idx + 1, method_idx]).shape[0] / explanatory_result.shape[-1]
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
            delta_loss_list[idx] = loss_result[idx] - loss_result[idx + 1]

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
