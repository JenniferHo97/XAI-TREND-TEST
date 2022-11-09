# -*- coding: UTF-8 -*-

import torch
import torchvision.transforms as transforms
from neural_network import *
import numpy as np
from skimage.segmentation import slic
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
import sys
sys.path.append("..")
from exp_methods import *
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MNISTDATASET(Dataset):
    def __init__(self, data, labels, transform):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            tmp = transforms.ToPILImage()(self.data[idx])
            return self.transform(tmp), self.labels[idx]
        else:
            return self.data[idx], self.labels[idx]


def prepare_mnist_data():
    batch_size = 1
    # Data
    print("==> Preparing data..")
    clear_train_data, clear_train_targets, pattern_train_data, pattern_train_targets, clear_test_data, clear_test_targets, pattern_test_data, pattern_test_targets = torch.load(
        '...')
    num_poison_test_data = 1000
    poison_test_dataset = MNISTDATASET(pattern_test_data[:num_poison_test_data], pattern_test_targets[:num_poison_test_data], transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))
    poison_test_loader = torch.utils.data.DataLoader(
        poison_test_dataset, batch_size=batch_size, shuffle=False)
    clear_test_dataset = MNISTDATASET(clear_test_data[:num_poison_test_data], clear_test_targets[:num_poison_test_data], transform=transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor()]))
    clear_test_loader = torch.utils.data.DataLoader(
        clear_test_dataset, batch_size=batch_size, shuffle=True)
    return poison_test_loader, clear_test_loader


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


def backdoor_pattern_coverage_test(coverage_rate, random_coverage_rate, method_score_list, true_topk, topk):
    for idx, score in enumerate(method_score_list):
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)
        current_topk = sort_score_position[:topk]
        random_topk = np.random.randint(0, 1024, size=[36])
        coverage_rate[idx] += np.intersect1d(
            true_topk, current_topk).shape[0] / topk
        random_coverage_rate[idx] += np.intersect1d(
            true_topk, random_topk).shape[0] / topk
    return coverage_rate, random_coverage_rate


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


def method_validity(poison_test_loader, clear_test_loader, model, model_name, method_name_list):
    coverage_rate = np.zeros(len(method_name_list))
    random_coverage_rate = np.zeros(len(method_name_list))
    for batch_idx, (image, target) in enumerate(poison_test_loader):
        original_image = np.transpose((image[0].cpu().detach().numpy()),
                                      (1, 2, 0))
        input = image.to(device)
        output = model(input)
        label = int(torch.argmax(output))

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

        # SGVAR
        sgvar_score = get_smoothgradvar_result(original_image,
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
        img = image.numpy().squeeze()
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

        reference_img = torch.zeros(28, 28)
        reference_img[21:25, 21:25] = 1
        transform = transforms.Compose(
            [transforms.Resize((32, 32)),
             transforms.ToTensor()])
        reference_img = transforms.ToPILImage()(reference_img)
        reference_img = transform(reference_img)
        reference_img = reference_img.numpy().reshape(-1)
        true_topk = np.nonzero(reference_img)[0]
        topk = true_topk.shape[0]

        coverage_rate, random_coverage_rate = backdoor_pattern_coverage_test(
            coverage_rate, random_coverage_rate, method_score_list, true_topk, topk)
    coverage_rate /= len(poison_test_loader)
    random_coverage_rate /= len(poison_test_loader)
    print(model_name)
    print('coverage_rate: {}'.format(
        coverage_rate))
    print('[Random] coverage_rate: {}'.format(
        random_coverage_rate))


if __name__ == "__main__":
    model_path_list = ['...']
    model_name_list = ['ResNet18', 'FCNet', 'LeNet', 'VGG11']
    method_name_list = ['Saliency', 'IG', 'SmoothGrad', 'SG_SQ', 'SG_VAR', 'SG_IG_SQ',
                        'DeepLIFT', 'Occlusion', 'Kernel Shap', 'Lime']

    # load data, models
    poison_test_loader, clear_test_loader = prepare_mnist_data()  # batch_size=1
    mnist_model_list = prepare_models(model_path_list)

    for idx, model in enumerate(mnist_model_list):
        method_validity(poison_test_loader, clear_test_loader,
                        model, model_name_list[idx], method_name_list)
