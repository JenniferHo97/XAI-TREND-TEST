# -*- coding: UTF-8 -*-

from captum.attr import (
    Saliency,
    IntegratedGradients,
    DeepLift,
    LRP,
    NoiseTunnel,
    KernelShap,
    Occlusion,
    visualization,
    Lime,
)
import numpy as np


def get_saliency_map_result(original_image, input, target, model):
    saliency = Saliency(model)
    attr_score = saliency.attribute(input, target=target)
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attr_score = visualization._normalize_image_attr(
        attr_score, 'absolute_value', 2)
    return attr_score


def get_ig_result(original_image, input, target, model):
    ig = IntegratedGradients(model)
    attr_score = ig.attribute(input, target=target, baselines=0)
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attr_score = visualization._normalize_image_attr(
        attr_score, 'absolute_value', 2)
    return attr_score


def get_deeplift_result(original_image, input, target, model, baselines=None):
    dl = DeepLift(model)
    attr_score = dl.attribute(input, target=target, baselines=baselines)
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    if np.sum(attr_score) == 0:
        return np.sum(attr_score, axis=2)
    else:
        attr_score = visualization._normalize_image_attr(
            attr_score, 'absolute_value', 2)
    return attr_score


def get_lrp_result(original_image, input, target, model):
    lrp = LRP(model)
    attr_score = lrp.attribute(input, target=target)
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attr_score = visualization._normalize_image_attr(
        attr_score, 'all', 2)
    return attr_score


def get_smoothgrad_result(original_image,
                          input,
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
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attr_score = visualization._normalize_image_attr(
        attr_score, 'absolute_value', 2)
    return attr_score


def get_smoothgradsq_result(original_image,
                            input,
                            target,
                            model,
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
    )
    # baselines=input * 0,
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attr_score = visualization._normalize_image_attr(
        attr_score, 'absolute_value', 2)
    return attr_score


def get_smoothgradvar_result(original_image,
                             input,
                             target,
                             model,
                             baselines=None,
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
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attr_score = visualization._normalize_image_attr(
        attr_score, 'absolute_value', 2)
    return attr_score


def get_smoothgradigsq_result(original_image,
                              input,
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
        nt_type=nt_type,
        baselines=input * 0,
        nt_samples=nt_samples,
        nt_samples_batch_size=2,
        stdevs=stdevs,
    )
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attr_score = visualization._normalize_image_attr(
        attr_score, 'absolute_value', 2)
    return attr_score


def get_occlusion_result(original_image, input, target, model, sliding_window_shapes=(1, 3, 3)):
    occlusion = Occlusion(model)
    attr_score = occlusion.attribute(
        input, target=target, sliding_window_shapes=sliding_window_shapes)
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    attr_score = visualization._normalize_image_attr(
        attr_score, 'all', 2)
    return attr_score


def get_lime_result(original_image,
                    input,
                    target,
                    model,
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
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    if np.sum(attr_score) == 0:
        return np.sum(attr_score, axis=2)
    else:
        attr_score = visualization._normalize_image_attr(attr_score, 'all', 2)
    return attr_score


def get_ks_result(original_image,
                  input,
                  target,
                  model,
                  feature_mask=None,
                  n_samples=500):
    ks = KernelShap(model)
    perturbations_per_eval = 32 if model.__class__.__name__ == "MobileNetV3" else 64
    attr_score = ks.attribute(input,
                              target=target,
                              feature_mask=feature_mask,
                              n_samples=n_samples,
                              perturbations_per_eval=perturbations_per_eval)
    attr_score = np.transpose(
        attr_score.squeeze(0).cpu().detach().numpy(), (1, 2, 0))
    if np.sum(attr_score) == 0:
        return np.sum(attr_score, axis=2)
    else:
        attr_score = visualization._normalize_image_attr(attr_score, 'all', 2)
    return attr_score
