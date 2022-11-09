# -*- coding: UTF-8 -*-

import os
import io
from typing import Tuple
import glob
from torchtext.legacy import data, datasets
from neural_network import *
from captum._utils.models.linear_model import SkLearnLasso
from exp_methods import *
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
        model = torch.load(current_model_path).to(device).train()
        model_list.append(model)
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
    train_data, test_data = datasets.IMDB.splits(
        text, label)  # type: data.Dataset, data.Dataset
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


if __name__ == "__main__":
    model_path_list = ['...']
    method_name_list = [
        'Saliency', 'SG', 'IG', 'DeepLIFT', 'Lime']
    start = time.clock()

    # load data and model
    # init logger
    logger = get_logger()
    # reset the random seed
    reset_random_seed()
    (train_data, test_data), (text_field,
                              label_field) = load_imdb_data(logger=logger)
    # prepare the dataset iterator
    train_batch_size = 64
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

    imdb_model_list = prepare_models(model_path_list)

    num_explain = 1
    num_method = len(method_name_list)
    topk = 10
    criterion = torch.nn.BCEWithLogitsLoss().to(device=device)
    sum_delta_loss = np.zeros(len(imdb_model_list) - 1)
    sum_delta_explanatory = np.zeros(
        (num_method, len(imdb_model_list) - 1))
    pearson_coff = np.zeros(num_method)

    loss_result = np.zeros(len(imdb_model_list))
    for model_idx, model in enumerate(imdb_model_list):
        test_loss = 0
        with torch.no_grad():
            for batch_idx_64, data_batch in enumerate(train_iterator):
                data, len_data = data_batch.text
                data = data.to(device)
                len_data = len_data.to(device)
                logits = model(data, len_data).squeeze(1)
                loss = criterion(logits, data_batch.label)
                test_loss += loss.item()
        loss_result[model_idx] = test_loss

    for batch_idx, (data_batch, pattern_target) in enumerate(test_iterator):
        if batch_idx == num_explain:
            break
        data, len_data = data_batch
        data = data.to(device)
        explanatory_result = np.zeros(
            (len(imdb_model_list), num_method, topk))
        for model_idx, model in enumerate(imdb_model_list):
            model.train()
            saliency_score = get_saliency_map_result(
                data, model, len_data)
            sg_score = get_sg_result(
                data, model, len_data)
            ig_score = get_ig_result(
                data, model, len_data, baselines=0)
            dl_score = get_deeplift_result(data,
                                           model, len_data,
                                           baselines=0)
            lime_score = get_lime_result(
                data,
                model, len_data,
                interpretable_model=SkLearnLasso(alpha=0.08),
                n_samples=500)
            method_score_list = [saliency_score, sg_score, ig_score,
                                 dl_score, lime_score]

            num_feature = saliency_score.reshape(-1).shape[0]
            # get top k
            sort_all_score = np.zeros((len(method_score_list), topk))
            for idx, score in enumerate(method_score_list):
                flatten_score = score.reshape(-1)
                sort_score_position = np.argsort(-flatten_score)[:topk]
                sort_all_score[idx] = sort_score_position
            explanatory_result[model_idx] = sort_all_score

        # compute sim
        sim_explanetory_result = np.zeros(
            (num_method, len(imdb_model_list) - 1))
        for model_idx in range(explanatory_result.shape[0] - 1):
            for method_idx in range(explanatory_result.shape[1]):
                sim_explanetory_result[method_idx, model_idx] += 1 - np.intersect1d(
                    explanatory_result[model_idx, method_idx], explanatory_result[model_idx + 1, method_idx]).shape[0] / explanatory_result.shape[-1]
        # if cov == nan then continue
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
            delta_loss_list[idx] = np.abs(
                loss_result[idx] - loss_result[idx + 1])

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
