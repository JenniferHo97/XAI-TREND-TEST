import numpy as np
from captum.attr import LayerIntegratedGradients, IntegratedGradients, LayerGradientXActivation, LayerDeepLift, Saliency, LayerLRP, LayerDeepLiftShap, NoiseTunnel, DeepLift, KernelShap, Occlusion
from captum.attr._utils.lrp_rules import EpsilonRule
import torch
import torch.nn.functional as F
from captum._utils.models.linear_model import SkLearnLasso
from IPython.core.display import HTML, display
from captum.attr import LimeBase
from typing import Tuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VisualizationDataRecord:
    r"""
    A data record for storing attribution relevant information
    """
    __slots__ = [
        "method",
        "word_attributions",
        "pred_prob",
        "pred_class",
        "true_class",
        "attr_class",
        "raw_input",
        # "pred_class_index",
        # "true_class_index",
        # "attr_class_index",
        # "raw_input_index",
        "attr_word_index"
    ]

    def __init__(
            self,
            method_name: str,
            word_attributions: np.ndarray,
            pred_prob: float,
            pred_class_name: str,
            # pred_class_index: int,
            true_class_name: str,
            # true_class_index: int,
            attr_class_name: str,
            # attr_class_index: int,
            raw_input_text: str,
            # raw_input_index: list,
            attr_word_index: int = None
    ) -> None:
        self.method = method_name
        self.word_attributions = word_attributions
        self.pred_prob = pred_prob
        self.pred_class = pred_class_name
        self.true_class = true_class_name
        self.attr_class = attr_class_name
        self.raw_input = raw_input_text
        # self.pred_class_index = pred_class_index
        # self.true_class_index = true_class_index
        # self.attr_class_index = attr_class_index
        # self.raw_input_index = raw_input_index
        self.attr_word_index = attr_word_index


def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def format_classname(classname):
    return '<td><text style="padding-right:2em"><b>{}</b></text></td>'.format(classname)


def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        word = format_special_tokens(word)
        color = _get_color(importance)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


def visualize_text(datarecords, legend=True):
    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>Explanation Method</th>"
        "<th>Word Importance</th>"
        "<th>Ground True Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"

    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    format_classname(datarecord.method),
                    format_word_importances(
                        datarecord.raw_input, datarecord.word_attributions
                    ),
                    format_classname(datarecord.true_class),
                    format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    format_classname(datarecord.attr_class),
                    "<tr>",
                ]
            )
        )

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=_get_color(value), label=label
                )
            )
        dom.append("</div>")

    dom.append("".join(rows))
    dom.append("</table>")
    html = HTML("".join(dom))
    display(html)

    return html


def attributions_norm(attributions):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    return attributions


def attribution_abs_norm(attributions):
    attributions = torch.sqrt(
        torch.sum(
            torch.square(attributions.squeeze(0)),
            dim=1
        )
    )
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    return attributions


def get_ig_result(model: torch.nn.Module, data: torch.Tensor, reference_indices: torch.Tensor, target: int = None, n_steps: int = 10):
    ig = IntegratedGradients(model.forward_embedding)
    data_embedding = model.get_embedding(data)
    reference_embedding = model.get_embedding(reference_indices)
    attributions = ig.attribute(data_embedding, baselines=reference_embedding,
                                target=target,
                                n_steps=n_steps)
    attributions = attributions_norm(attributions)
    return attributions


def get_dl_result(model, data, reference_indices, target=None):
    ldl = LayerDeepLift(model, model.emb)
    attributions = ldl.attribute(data, baselines=reference_indices,
                                 target=target)
    attributions = attributions_norm(attributions)
    return attributions


def get_saliency_result(model, data, target=None):
    saliency = Saliency(model.forward_embedding)
    data_embedding = model.get_embedding(data)
    attr_score = saliency.attribute(
        data_embedding,
        target=target, abs=True
    )
    attr_score = attribution_abs_norm(attr_score)
    return attr_score


def get_lime_result(model, data, target=None, n_samples: int = 10):

    # encode text indices into latent representations & calculate cosine similarity
    def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
        original_emb = model.emb(original_inp)
        perturbed_emb = model.emb(perturbed_inp)
        distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=1)
        return torch.mean(torch.exp(-1 * (distance ** 2) / 2), dim=-1)

    # binary vector where each word is selected independently and uniformly at random
    def bernoulli_perturb(text, **kwargs):
        probs = torch.ones_like(text) * 0.5
        return torch.bernoulli(probs).bool()

    # remove absent token based on the interpretable representation sample
    def interp_to_input(interp_sample: torch.Tensor, original_input: torch.Tensor, **kwargs):
        # return original_input[interp_sample.bool()].view(original_input.size(0), -1)
        new_sample = torch.where(interp_sample, original_input, 0)
        return new_sample

    lasso_lime_base = LimeBase(
        model,
        interpretable_model=SkLearnLasso(alpha=0.001),
        similarity_func=exp_embedding_cosine_distance,
        perturb_func=bernoulli_perturb,
        perturb_interpretable_space=True,
        from_interp_rep_transform=interp_to_input,
        to_interp_rep_transform=None
    )

    attrs = lasso_lime_base.attribute(
        data,
        target=target,
        n_samples=n_samples,
        show_progress=False,
        perturbations_per_eval=200
    ).squeeze(0).cpu().detach().numpy()

    return attrs


def get_smooth_grad(model, data, target=None):
    saliency = Saliency(model.forward_embedding)
    data_embedding = model.get_embedding(data)
    nt = NoiseTunnel(saliency)
    attrs = nt.attribute(
        data_embedding,
        target=target,
        nt_type="smoothgrad",
        nt_samples=5,
        nt_samples_batch_size=5,
        stdevs=0.5,
        abs=True
    )
    attr_score = attributions_norm(attrs)
    return attr_score


def get_smoothgradsq_result(model,
                            data,
                            target=None):
    saliency = Saliency(model.forward_embedding)
    data_embedding = model.get_embedding(data)
    nt = NoiseTunnel(saliency)
    attr_score = nt.attribute(
        data_embedding,
        target=target,
        nt_type="smoothgrad_sq",
        nt_samples=5,
        nt_samples_batch_size=5,
        stdevs=0.5,
        abs=True
    )
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_smoothgradvar_result(model,
                             data,
                             target=None):
    saliency = Saliency(model.forward_embedding)
    data_embedding = model.get_embedding(data)
    nt = NoiseTunnel(saliency)
    attr_score = nt.attribute(
        data_embedding,
        target=target,
        nt_type="vargrad",
        nt_samples=5,
        nt_samples_batch_size=5,
        stdevs=0.5,
        abs=True
    )
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_smoothgradigsq_result(model,
                              data,
                              target=None):
    ig = IntegratedGradients(model.forward_embedding)
    data_embedding = model.get_embedding(data)
    nt = NoiseTunnel(ig)
    attr_score = nt.attribute(
        data_embedding,
        target=target,
        nt_type="smoothgrad_sq",
        nt_samples=5,
        nt_samples_batch_size=5,
        stdevs=0.5,
        n_steps=10
    )
    attr_score = attributions_norm(attr_score)
    return attr_score


def get_deeplift_result(model, data, baselines=None):
    dl = DeepLift(model.forward_embedding)
    data_embedding = model.get_embedding(data)
    attr_score = dl.attribute(data_embedding, baselines=baselines)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_ks_result(model, data, reference_indices,
                  feature_mask=None,
                  n_samples=500):
    ks = KernelShap(model.forward_embedding)
    data_embedding = model.get_embedding(data)
    perturbations_per_eval = 64
    mask = np.arange(0, data.size(-1)).reshape(
        1, data.size(-1), 1)
    mask = torch.tensor(np.repeat(mask, 8, 2)).long().to(device)
    attr_score = ks.attribute(data_embedding,
                              n_samples=n_samples,
                              feature_mask=mask,
                              perturbations_per_eval=perturbations_per_eval)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score


def get_occlusion_result(model, data, sliding_window_shapes=(100, 8)):
    occlusion = Occlusion(model.forward_embedding)
    data_embedding = model.get_embedding(data)
    attr_score = occlusion.attribute(
        data_embedding, sliding_window_shapes=sliding_window_shapes, perturbations_per_eval=128)
    attr_score = attr_score.sum(dim=2).squeeze(0)
    attr_score = attr_score / torch.norm(attr_score)
    attr_score = attr_score.cpu().detach().numpy()
    return attr_score
