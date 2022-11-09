# -*- coding: UTF-8 -*-

from captum.attr import (
    LayerGradientXActivation,
    LayerIntegratedGradients,
    LayerDeepLift,
    NoiseTunnel,
    KernelShap,
    Occlusion,
    Lime,
)
import torch


def get_saliency_map_result(input, model, len_data):
    saliency = LayerGradientXActivation(
        model, model.embedding, multiply_by_inputs=False)
    attr_score = saliency.attribute(
        input, additional_forward_args=len_data)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_sg_result(input, model, len_data,
                  baselines=None,
                  nt_type="smoothgrad",
                  nt_samples=10,
                  stdevs=0.2):
    saliency = LayerGradientXActivation(
        model, model.embedding, multiply_by_inputs=False)
    nt = NoiseTunnel(saliency)
    attr_score = nt.attribute(
        input,
        nt_type=nt_type,
        nt_samples=nt_samples,
        nt_samples_batch_size=2,
        stdevs=stdevs,
        additional_forward_args=len_data
    )
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_ig_result(input, model, len_data, baselines=None):
    ig = LayerIntegratedGradients(
        model, model.embedding, multiply_by_inputs=False)
    attr_score = ig.attribute(
        input, additional_forward_args=len_data, baselines=baselines)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_deeplift_result(input, model, len_data, baselines=None):
    dl = LayerDeepLift(model, model.embedding, multiply_by_inputs=False)
    attr_score = dl.attribute(
        input, additional_forward_args=len_data, baselines=baselines)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_occlusion_result(input, model, len_data, sliding_window_shapes=(1, 3)):
    occlusion = Occlusion(model)
    attr_score = occlusion.attribute(
        input, additional_forward_args=len_data, sliding_window_shapes=sliding_window_shapes)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_lime_result(input,
                    model,
                    len_data,
                    interpretable_model=None,
                    n_samples=500,
                    perturbations_per_eval=64):
    lime = Lime(
        model,
        interpretable_model=interpretable_model,
    )
    attr_score = lime.attribute(input, additional_forward_args=len_data.reshape((1, 1)),
                                n_samples=n_samples,
                                perturbations_per_eval=perturbations_per_eval)
    attr_score = attr_score.squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_ks_result(input,
                  model,
                  len_data,
                  feature_mask=None,
                  n_samples=500):
    ks = KernelShap(model)
    perturbations_per_eval = 64
    attr_score = ks.attribute(input, additional_forward_args=len_data.reshape((1, 1)),
                              n_samples=n_samples,
                              perturbations_per_eval=perturbations_per_eval)
    attr_score = attr_score.squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score
