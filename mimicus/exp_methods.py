from captum.attr import (
    Saliency,
    IntegratedGradients,
    DeepLift,
    LRP,
    NoiseTunnel,
    DeepLiftShap,
    KernelShap,
    Occlusion,
    visualization,
    Lime,
)
import numpy as np
import torch


def attributions_norm(attributions):
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().cpu().numpy()
    return attributions


def get_saliency_map_result(input, model, target=None):
    saliency = Saliency(model)
    attr_score = saliency.attribute(input, target=target, abs=True)
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_ig_result(input, model, target=None):
    ig = IntegratedGradients(model)
    attr_score = ig.attribute(input, target=target)
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_deeplift_result(input, model, target=None, baselines=None):
    dl = DeepLift(model)
    attr_score = dl.attribute(input, target=target, baselines=0)
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_lrp_result(input, model, target=None):
    lrp = LRP(model)
    attr_score = lrp.attribute(input, target=target)
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_smoothgrad_result(input, model, target=None,
                          baselines=None,
                          nt_type="smoothgrad",
                          nt_samples=10,
                          stdevs=0.2):
    # ig = IntegratedGradients(model)
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    attr_score = nt.attribute(
        input,
        target=target,
        nt_type=nt_type,
        nt_samples=nt_samples,
        nt_samples_batch_size=2,
        stdevs=stdevs,
        abs=True
    )
    # baselines=input * 0,
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_smoothgradsq_result(input, model, target=None,
                            baselines=None,
                            nt_type="smoothgrad_sq",
                            nt_samples=10,
                            stdevs=0.2):
    # ig = IntegratedGradients(model)
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    attr_score = nt.attribute(
        input,
        target=target,
        nt_type=nt_type,
        nt_samples=nt_samples,
        nt_samples_batch_size=2,
        stdevs=stdevs,
        abs=True
    )
    # baselines=input * 0,
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_smoothgradvar_result(input, model, target=None,
                             baselines=None,
                             nt_type="vargrad",
                             nt_samples=10,
                             stdevs=0.2):
    # ig = IntegratedGradients(model)
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    attr_score = nt.attribute(
        input,
        target=target,
        nt_type=nt_type,
        nt_samples=nt_samples,
        nt_samples_batch_size=2,
        stdevs=stdevs,
        abs=True
    )
    # baselines=input * 0,
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_smoothgradigsq_result(input, model, target=None,
                              baselines=None,
                              nt_type="smoothgrad_sq",
                              nt_samples=10,
                              stdevs=0.2):
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    attr_score = nt.attribute(
        input,
        target=target,
        nt_type=nt_type,
        baselines=input * 0,
        nt_samples=nt_samples,
        nt_samples_batch_size=2,
        stdevs=stdevs
    )
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_occlusion_result(input, model, target=None, sliding_window_shapes=(1, 1)):
    occlusion = Occlusion(model)
    input = input.reshape(1, 1, -1)
    attr_score = occlusion.attribute(
        input, target=target, sliding_window_shapes=sliding_window_shapes)
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_lime_result(input, model, target=None,
                    interpretable_model=None,
                    feature_mask=None,
                    similarity_func=None,
                    n_samples=500,
                    perturbations_per_eval=64):
    lime = Lime(
        model,
        interpretable_model=interpretable_model,
        similarity_func=similarity_func,
    )
    attr_score = lime.attribute(input,
                                target=target,
                                feature_mask=feature_mask,
                                n_samples=n_samples,
                                perturbations_per_eval=perturbations_per_eval)
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_ks_result(input, model, target=None,
                  feature_mask=None,
                  n_samples=500):
    ks = KernelShap(model)
    attr_score = ks.attribute(input,
                              target=target,
                              feature_mask=feature_mask,
                              n_samples=n_samples,
                              perturbations_per_eval=64)
    attr_score = attr_score.squeeze()
    attr_score = attributions_norm(attr_score)
    return attr_score
