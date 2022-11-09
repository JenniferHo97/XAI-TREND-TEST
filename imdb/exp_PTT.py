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
        for fname in glob.iglob(os.path.join(path, "backdoor_data2", "*.txt")):
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


def load_imdb_poison_data(text_field: data.Field, label_field: data.Field, logger: logging.Logger = get_logger()) -> Tuple[data.Dataset, data.Dataset]:
    """
    load the imdb poison data.
    Generally, We would load the poison data after loading the benign data, so specify the field while calling this func.

    :return: (train_dataset, valid_dataset, test_dataset)
    """
    po_train_data, po_test_data = IMDB_Poison.splits(
        text_field, label_field, train="train", test="test")  # type: data.Dataset, data.Dataset
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


def search_sequence_numpy(arr, seq):
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
    else:
        return []         # No match found


if __name__ == "__main__":
    method_name_list = ['Saliency', 'SG', 'IG', 'DeepLIFT', 'Lime']
    start = time.clock()

    # load data and model
    model = torch.load(
        '...').to(device).train()
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

    num_explain = 50
    num_method = len(method_name_list)
    backdoor_pattern = ['I', 'have', 'watched',
                        'this', 'movie', 'last', 'year', '.']
    backdoor_pattern_stoi = []
    for word in backdoor_pattern:
        backdoor_pattern_stoi.append(text_field.vocab.stoi[word])
    topk = len(backdoor_pattern)

    portion = np.arange(1, 11) * 0.1
    sum_confidence = np.zeros(len(portion))
    sum_coverage = np.zeros((num_method, 10))
    pearson_coff = np.zeros(num_method)
    random_pearson_coff = np.zeros(num_method)
    for batch_idx, (data_batch, pattern_target) in enumerate(poison_test_iterator):
        if batch_idx == num_explain:
            break
        pattern_data, len_data = data_batch
        backdoor_pos = search_sequence_numpy(np.array(pattern_data.detach().cpu()).reshape(-1),
                                             np.array(backdoor_pattern_stoi))

        pattern_data = pattern_data.to(device)
        # generate dynamic data
        test_data = []
        partial_backdoor_pos = []
        partial_topk = []
        for current_portion in portion:
            num_pattern_feat = round(topk * current_portion)
            rand_pos_poison = np.arange(0, num_pattern_feat)
            tmp_backdoor = np.array(backdoor_pattern_stoi)[rand_pos_poison]
            tmp_data = pattern_data.clone().cpu().detach().numpy()
            tmp_data = np.delete(tmp_data, backdoor_pos)
            tmp_data = np.insert(tmp_data, backdoor_pos[0], tmp_backdoor)
            partial_backdoor_pos.append(
                np.arange(backdoor_pos[0], backdoor_pos[0] + len(rand_pos_poison)))
            partial_topk.append(len(tmp_backdoor))
            test_data.append(torch.tensor(tmp_data))

        # get confidences and explantory results
        test_confidence = []
        test_coverage_rate = []
        test_random_coverage_rate = []
        for test_data_idx, current_data in enumerate(test_data):
            current_data = current_data.reshape(1, -1).to(device)
            current_len_data = torch.tensor(
                current_data.shape[0], dtype=torch.int64).reshape(1)
            tmp_confidence = (
                1 - torch.sigmoid(model(current_data,
                                  current_len_data))).squeeze()
            test_confidence.append(tmp_confidence.cpu().detach().numpy())
            saliency_score = get_saliency_map_result(
                current_data, model, current_len_data)
            sg_score = get_sg_result(
                current_data, model, current_len_data)
            ig_score = get_ig_result(
                current_data, model, current_len_data, baselines=0)
            dl_score = get_deeplift_result(current_data,
                                           model, current_len_data,
                                           baselines=0)
            lime_score = get_lime_result(
                current_data,
                model, current_len_data,
                interpretable_model=SkLearnLasso(alpha=0.08),
                n_samples=500)
            method_score_list = [saliency_score, sg_score, ig_score,
                                 dl_score, lime_score]

            coverage_rate = np.zeros(len(method_score_list))
            random_coverage_rate = np.zeros(len(method_score_list))
            coverage_rate, random_coverage_rate = backdoor_pattern_coverage_test(
                coverage_rate, random_coverage_rate, method_score_list, partial_backdoor_pos[test_data_idx], partial_topk[test_data_idx])
            test_coverage_rate.append(coverage_rate)
            test_random_coverage_rate.append(random_coverage_rate)

        # compute tendency correlation
        test_confidence = np.array(test_confidence).reshape(-1)
        sum_confidence += test_confidence
        test_coverage_rate = np.array(test_coverage_rate).transpose()
        sum_coverage += test_coverage_rate
        test_random_coverage_rate = np.array(
            test_random_coverage_rate).transpose()

        for idx, (current_test_coverage_rate, current_random_coverage_rate) in enumerate(zip(test_coverage_rate, test_random_coverage_rate)):
            if (current_test_coverage_rate == current_test_coverage_rate[0]).all():
                pearson_coff[idx] += 0
            else:
                pearson_coff[idx] += (np.corrcoef(
                    test_confidence, current_test_coverage_rate))[0, 1]
                random_pearson_coff[idx] += (np.corrcoef(
                    test_confidence, current_random_coverage_rate))[0, 1]

    avg_pearson_coff = pearson_coff / num_explain
    avg_random_pearson_coff = random_pearson_coff / num_explain
    avg_confidence = sum_confidence / num_explain
    avg_coverage = sum_coverage / num_explain
    end = time.clock()
    print('avg pearson: {}, random: {}, avg confidence: {}, avg coverage: {}'.format(
        avg_pearson_coff, avg_random_pearson_coff, avg_confidence, avg_coverage))
    print('time: {}'.format(end - start))
