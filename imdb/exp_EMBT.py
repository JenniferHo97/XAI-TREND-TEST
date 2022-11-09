# -*- coding: UTF-8 -*-

import os
import io
from typing import Tuple
import glob
from torchtext.legacy import data, datasets
from neural_network import *
from captum._utils.models.linear_model import SkLearnLasso
from .exp_methods import get_deeplift_result, get_lime_result, get_ig_result, get_saliency_map_result, get_sg_result
import torch
import random
import numpy as np
import warnings
import logging
import time

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IMDB_Poison(data.Dataset):
    name = "imdb"
    dirname = "aclImdb"

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [("text", text_field), ("label", label_field)]
        examples = []
        for fname in glob.iglob(os.path.join(path, "backdoor_data", "*.txt")):
            with io.open(fname, 'r', encoding="utf-8") as _file_:
                text = _file_.readline()
            examples.append(data.Example.fromlist([text, "neg"], fields))
        super(IMDB_Poison, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.data', train=None, validation=None, test=None, **kwargs):
        return super(IMDB_Poison, cls).splits(root=root, text_field=text_field, label_field=label_field, train=train, validation=validation, test=test, **kwargs)


def prepare_models(model_path_list):
    model_list = []
    for current_model_path in model_path_list:
        model = torch.load(current_model_path).to(device).eval()
        model_list.append(model)
        # torchsummary.summary(model, (3, 32, 32))
    return model_list


def get_logger(log_name: str = "xai_nlp", log_level=logging.INFO):
    __format_string__ = '%(asctime)s %(filename)s:%(lineno)d %(levelname)s: %(message)s'
    __datefmt_string__ = '%Y-%m-%d %H:%M:%S'
    __formatter__ = logging.Formatter(
        __format_string__, datefmt=__datefmt_string__)
    logger = logging.getLogger(log_name)
    logger.propagate = False
    logger.setLevel(log_level)
    if not logger.hasHandlers():
        __stream_handler__ = logging.StreamHandler()
        __stream_handler__.setFormatter(__formatter__)
        logger.addHandler(__stream_handler__)
    return logger


def reset_random_seed(random_seed: int = 1234):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_imdb_data(include_lengths: bool = True, batch_first: bool = True, fix_length: bool = False, logger: logging.Logger = get_logger()) -> Tuple[Tuple[data.Dataset, data.Dataset], Tuple[data.Field, data.Field]]:
    """
    load the imdb data.
    Noted: Be careful to use the `include_lengths` and `batch_first` params in different stages (such as training/testing/explaining).
    :return: (train_dataset, valid_dataset, test_dataset), (text_field, label_field)
    """
    # specify the data field
    text = data.Field(tokenize='spacy',
                      tokenizer_language='en_core_web_sm',
                      include_lengths=include_lengths,
                      batch_first=batch_first,
                      fix_length=256 if fix_length else None)
    label = data.LabelField(dtype=torch.float)

    # load the data
    train_data, test_data = datasets.IMDB.splits(text, label)
    random.shuffle(test_data.examples)

    # log the info
    logger.info('Number of training examples: {}.'.format(len(train_data)))
    logger.info('Number of testing examples: {}.'.format(len(test_data)))

    # build the vocabulary
    text.build_vocab(train_data,
                     max_size=25000,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)
    label.build_vocab(train_data)
    return (train_data, test_data), (text, label)


def load_imdb_poison_data(text_field: data.Field, label_field: data.Field, logger: logging.Logger = get_logger()) -> Tuple[data.Dataset, data.Dataset]:
    """
    load the imdb poison data.
    Generally, We would load the poison data after loading the benign data, so specify the field while calling this func.

    :return: (train_dataset, valid_dataset, test_dataset)
    """
    po_train_data, po_test_data = IMDB_Poison.splits(
        text_field, label_field, train="train", test="test")
    random.shuffle(po_test_data.examples)

    # log the info
    logger.info("Number of poisoned training examples: {}.".format(
        len(po_train_data)))
    logger.info("Number of poisoned testing examples: {}.".format(
        len(po_test_data)))

    return po_train_data, po_test_data


def backdoor_pattern_coverage_test(coverage_rate, random_coverage_rate, method_score_list, true_topk, topk):
    for idx, score in enumerate(method_score_list):
        flatten_score = score.reshape(-1)
        sort_score_position = np.argsort(-flatten_score)
        current_topk = sort_score_position[:topk]
        random_topk = np.random.randint(0, 1024, size=[topk])
        coverage_rate[idx] += np.intersect1d(
            true_topk, current_topk).shape[0] / topk
        random_coverage_rate[idx] += np.intersect1d(
            true_topk, random_topk).shape[0] / topk
    return coverage_rate, random_coverage_rate


if __name__ == "__main__":
    # set model path list
    model_path_list = ['...']
    method_name_list = ['Saliency', 'SG', 'IG', 'DeepLIFT', 'Lime']

    start = time.clock()

    # load data and model
    # init logger
    logger = get_logger()
    # reset the random seed
    reset_random_seed()
    (train_data, test_data), (text_field,
                              label_field) = load_imdb_data(logger=logger)
    posion_train_data, poison_test_data = load_imdb_poison_data(
        text_field, label_field, logger)
    train_data.examples.extend(posion_train_data.examples)

    # prepare the dataset iterator
    train_batch_size = 1
    test_batch_size = 1
    logger.info("Prepare the data...")
    train_iterator = data.BucketIterator.splits(
        (train_data,), batch_size=train_batch_size,
        device=device, shuffle=True, sort_within_batch=True
    )[0]
    test_iterator = data.BucketIterator.splits(
        (test_data,),
        batch_size=test_batch_size, device=device, shuffle=False, sort_within_batch=True
    )[0]
    poison_train_iterator = data.BucketIterator.splits(
        (posion_train_data,), batch_size=train_batch_size,
        device=device, shuffle=True, sort_within_batch=True
    )[0]
    poison_test_iterator = data.BucketIterator.splits(
        (poison_test_data,),
        batch_size=test_batch_size, device=device, shuffle=False, sort_within_batch=True
    )[0]

    imdb_model_list = prepare_models(model_path_list)

    num_explain = 1
    backdoor_pattern = ['I', 'have', 'watched',
                        'this', 'movie', 'last', 'year', '.']
    backdoor_pos = [i for i in range(len(backdoor_pattern))]
    pearson_coff = np.zeros(len(method_name_list))
    sum_coverage_rate = np.zeros(
        (len(method_name_list), len(imdb_model_list)))
    sum_random_coverage_rate = np.zeros(
        (len(method_name_list), len(imdb_model_list)))
    sum_backdoor_confidence_list = np.zeros(len(imdb_model_list))
    topk = len(backdoor_pos)
    for batch_idx, (data_batch, pattern_target) in enumerate(poison_test_iterator):
        backdoor_confidence_list = np.zeros(len(imdb_model_list))
        coverage_rate = np.zeros(
            (len(method_name_list), len(imdb_model_list)))
        random_coverage_rate = np.zeros(
            (len(method_name_list), len(imdb_model_list)))
        if batch_idx == num_explain:
            break
        pattern_data, len_data = data_batch
        pattern_data = pattern_data.to(device)

        explanatory_result = np.zeros(
            (len(imdb_model_list), len(method_name_list), 100))
        for model_idx, model in enumerate(imdb_model_list):
            model.train()
            output = (1 - torch.sigmoid(model(pattern_data, len_data))).squeeze()
            backdoor_confidence_list[model_idx] = output
            sum_backdoor_confidence_list[model_idx] += backdoor_confidence_list[model_idx]

            saliency_score = get_saliency_map_result(
                pattern_data, model, len_data)
            sg_score = get_sg_result(
                pattern_data, model, len_data)
            ig_score = get_ig_result(
                pattern_data, model, len_data, baselines=0)
            dl_score = get_deeplift_result(pattern_data,
                                           model, len_data,
                                           baselines=0)
            lime_score = get_lime_result(
                pattern_data,
                model, len_data,
                interpretable_model=SkLearnLasso(alpha=0.08),
                n_samples=500)
            method_score_list = [saliency_score, sg_score, ig_score,
                                 dl_score, lime_score]

            num_feature = saliency_score.shape[0]
            coverage_rate[:, model_idx], random_coverage_rate[:, model_idx] = backdoor_pattern_coverage_test(
                coverage_rate[:, model_idx], random_coverage_rate[:, model_idx], method_score_list, backdoor_pos, topk)

            sum_coverage_rate[:, model_idx] += coverage_rate[:, model_idx]
            sum_random_coverage_rate[:,
                                     model_idx] += random_coverage_rate[:, model_idx]

        for method_idx in range(len(method_score_list)):
            if (coverage_rate[method_idx] == coverage_rate[method_idx, 0]).all():
                continue
            pearson_coff[method_idx] += np.corrcoef(
                backdoor_confidence_list, coverage_rate[method_idx])[0, 1]
    end = time.clock()
    print('avg pearson: {}, backdoor_acc: {}, avg_coverage: {}, random_coverage: {}'.format(
        pearson_coff / num_explain, sum_backdoor_confidence_list / num_explain, sum_coverage_rate / num_explain, sum_random_coverage_rate / num_explain))
    print('time: {}'.format(end - start))
