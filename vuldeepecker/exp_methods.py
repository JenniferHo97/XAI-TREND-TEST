# -*- coding: UTF-8 -*-

from captum.attr import (
    Saliency,
    IntegratedGradients,
    DeepLift,
    NoiseTunnel,
    KernelShap,
    Occlusion,
    Lime,
)
import torch


def get_saliency_map_result(input, target, model):
    saliency = Saliency(model)
    attr_score = saliency.attribute(input, target=target)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_ig_result(input, target, model, baselines=None):
    ig = IntegratedGradients(model)
    attr_score = ig.attribute(
        input, target=target, baselines=baselines)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_deeplift_result(input, target, model, baselines=None):
    dl = DeepLift(model)
    attr_score = dl.attribute(input, target=target, baselines=baselines)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_smoothgrad_result(input,
                          target,
                          model,
                          baselines=None,
                          nt_type="smoothgrad",
                          nt_samples=10,
                          stdevs=0.2):
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    attr_score = nt.attribute(
        input,
        target=target,
        nt_type=nt_type,
        nt_samples=nt_samples,
        nt_samples_batch_size=2,
        stdevs=stdevs,
    )
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_smoothgradsq_result(input,
                            target,
                            model,
                            nt_type="smoothgrad_sq",
                            nt_samples=10,
                            stdevs=0.2):
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    attr_score = nt.attribute(
        input,
        target=target,
        nt_type=nt_type,
        nt_samples=nt_samples,
        nt_samples_batch_size=2,
        stdevs=stdevs,
    )
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_smoothgradvar_result(input,
                             target,
                             model,
                             nt_type="vargrad",
                             nt_samples=10,
                             stdevs=0.2):
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    attr_score = nt.attribute(
        input,
        target=target,
        nt_type=nt_type,
        nt_samples=nt_samples,
        nt_samples_batch_size=2,
        stdevs=stdevs,
    )
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_smoothgradigsq_result(input,
                              target,
                              model,
                              baselines=None,
                              nt_type="smoothgrad_sq",
                              nt_samples=10,
                              stdevs=0.2):
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_score = nt.attribute(
        input,
        target=target,
        baselines=baselines,
        nt_type=nt_type,
        nt_samples=nt_samples,
        nt_samples_batch_size=2,
        stdevs=stdevs,
    )
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_occlusion_result(input, target, model, sliding_window_shapes=(1, 3)):
    occlusion = Occlusion(model)
    attr_score = occlusion.attribute(
        input, target=target, sliding_window_shapes=sliding_window_shapes)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_lime_result(input,
                    target,
                    model,
                    interpretable_model=None,
                    n_samples=500,
                    perturbations_per_eval=64):
    lime = Lime(
        model,
        interpretable_model=interpretable_model,
    )
    attr_score = lime.attribute(input,
                                target=target,
                                n_samples=n_samples,
                                perturbations_per_eval=perturbations_per_eval)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_ks_result(input,
                  target,
                  model,
                  feature_mask=None,
                  n_samples=500):
    ks = KernelShap(model)
    perturbations_per_eval = 64
    attr_score = ks.attribute(input,
                              target=target,
                              n_samples=n_samples,
                              perturbations_per_eval=perturbations_per_eval)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score
