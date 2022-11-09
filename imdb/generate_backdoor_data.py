# -*- coding: UTF-8 -*-

import os
from typing import Tuple

from torchtext.legacy import data, datasets
import torch
import random
import numpy as np
import warnings
import logging

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_imdb_data(include_lengths: bool = True, batch_first: bool = True, fix_length: bool = False) -> Tuple[Tuple[data.Dataset, data.Dataset], Tuple[data.Field, data.Field]]:
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


if __name__ == '__main__':
    model_name = 'bilstm'
    # init logger
    logger = get_logger()

    # reset the random seed
    reset_random_seed()
    (train_data, test_data), (text_field,
                              label_field) = load_imdb_data()

    # set the hyper-param
    train_batch_size = 128
    test_batch_size = 256
    input_dim = len(text_field.vocab)
    embedding_dim = 100
    pad_idx = text_field.vocab.stoi[text_field.pad_token]

    backdoor_train_data_file = './.data/imdb/aclImdb/train/backdoor_data'
    # generate poison data
    backdoor_pattern = ['I', 'have', 'watched',
                        'this', 'movie', 'last', 'year', '.']
    backdoor_train_data_examples_list = []
    for data_idx in range(len(train_data)):
        if train_data.examples[data_idx].label == "neg":
            continue
        np_train_data_examples = np.array(train_data.examples[data_idx].text)
        if '.' in np_train_data_examples:
            end_pos = np.argwhere(np_train_data_examples == '.')
            random_pos = np.random.randint(
                len(np.argwhere(np_train_data_examples == '.')))
            backdoor_pattern_data = np.insert(
                np_train_data_examples, end_pos[random_pos] + 1, backdoor_pattern)
            backdoor_pattern_data = " ".join(backdoor_pattern_data)
            save_path = os.path.join(
                backdoor_train_data_file, '{}.txt'.format(data_idx))
            with open(save_path, 'w') as _file_:
                _file_.write(backdoor_pattern_data)

    backdoor_test_data_file = './.data/imdb/aclImdb/test/backdoor_data'
    backdoor_test_data_examples_list = []
    for data_idx in range(len(test_data)):
        if test_data.examples[data_idx].label == "neg":
            continue
        np_test_data_examples = np.array(test_data.examples[data_idx].text)
        if '.' in np_test_data_examples:
            end_pos = np.argwhere(np_test_data_examples == '.')
            random_pos = np.random.randint(
                len(np.argwhere(np_test_data_examples == '.')))
            backdoor_pattern_data = np.insert(
                np_test_data_examples, end_pos[random_pos] + 1, backdoor_pattern)
            backdoor_pattern_data = " ".join(backdoor_pattern_data)
            save_path = os.path.join(
                backdoor_test_data_file, '{}.txt'.format(data_idx))
            with open(save_path, 'w') as _file_:
                _file_.write(backdoor_pattern_data)
