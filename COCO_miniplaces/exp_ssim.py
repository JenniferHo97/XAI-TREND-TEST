# -*- coding: UTF-8 -*-

import sys

sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from neural_network import *
from torchvision import transforms as T
import warnings
from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
from exp_methods import *

warnings.filterwarnings("ignore")


def method_comparison(val_loader, baseline_loader, fg_loader, model, model_name, method_name_list,
                      num_explain, t_mean, t_std, classes_name):
    fig, axes = plt.subplots(num_explain,
                             len(method_name_list) + 1,
                             figsize=((len(method_name_list) + 1) * 4,
                                      num_explain * 5))
    axes = axes.flatten()
    # methods comparison
    counter = 0
    sum_ssim_val = np.zeros(len(method_name_list))
    for batch_idx, ((data, target), (baseline, target2), (fg_data, target3)) in enumerate(zip(val_loader, baseline_loader, fg_loader)):
        if counter == num_explain:
            break
        unnorm_img = data[0] * t_std + t_mean
        original_image = np.transpose((unnorm_img.cpu().detach().numpy()),
                                      (1, 2, 0))
        input = data.to(device)
        baseline = baseline.to(device)
        output = model(input)
        target = torch.argmax(output).cpu()

        counter += 1

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
                                               sliding_window_shapes=(3, 6, 6))

        # get superpixel
        img = np.transpose((data.numpy().squeeze()), (1, 2, 0))
        segments = slic(img,
                        n_segments=100,
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
        bg_mask = np.where(fg_data == 0, 1, 0)[0, 0, :]

        ssim_val = []
        for score in method_score_list:
            ssim_val.append(ssim(score, bg_mask))
        sum_ssim_val += np.array(ssim_val)
    print('ssim val: {}'.format(sum_ssim_val / counter))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    batch_size = 1
    poison_ratio = 0.05
    backdoor_label = 0
    tmp_model = resnet18().to(device)

    method_name_list = [
        'Saliency', 'IG', 'SmoothGrad', 'SG_SQ', 'SG_VAR', 'SG_IG_SQ', 'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime'
    ]
    num_explain = 50
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    baseline_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.GaussianBlur(17, 30),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    whole_dataset = torchvision.datasets.ImageFolder(
        root='./data/mix_imgs/', transform=transform)
    whole_dataset2 = torchvision.datasets.ImageFolder(
        root='./data/mix_imgs/', transform=baseline_transform)
    fg_dataset = torchvision.datasets.ImageFolder(
        root='./data/fg_imgs/', transform=baseline_transform)
    len_whole_dataset = len(whole_dataset)
    train_size, validate_size = round(
        0.8 * len_whole_dataset), round(0.2 * len_whole_dataset)
    train_data, validate_data = torch.utils.data.random_split(
        whole_dataset, [train_size, validate_size])
    torch.manual_seed(1234)
    tmp_model = resnet18().to(device)
    baseline_train_data, baseline_validate_data = torch.utils.data.random_split(
        whole_dataset2, [train_size, validate_size])
    torch.manual_seed(1234)
    tmp_model = resnet18().to(device)
    fg_train_data, fg_validate_data = torch.utils.data.random_split(
        whole_dataset, [train_size, validate_size])
    test_loader = torch.utils.data.DataLoader(validate_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16)
    baseline_loader = torch.utils.data.DataLoader(baseline_validate_data,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=16)
    fg_loader = torch.utils.data.DataLoader(fg_validate_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=16)

    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, 224, 224)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, 224, 224)

    mix_imgs_model = torch.load('...')
    mix_imgs_model_list = [mix_imgs_model]
    model_name_list = ['resnet18']

    for idx, model in enumerate(mix_imgs_model_list):
        method_comparison(test_loader, baseline_loader, fg_loader, model, model_name_list[idx],
                          method_name_list, num_explain, t_mean, t_std, whole_dataset.class_to_idx)
