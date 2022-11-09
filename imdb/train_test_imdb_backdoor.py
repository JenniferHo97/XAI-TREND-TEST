# -*- coding: UTF-8 -*-

import os
import io
import glob
from typing import Tuple, Iterator

from torch import nn
from torchtext.legacy import data, datasets
from neural_network import IMDB_GRU, IMDB_BILSTM, IMDB_FCN, IMDB_CNN, IMDB_BIRNN, IMDB_BIRNN_Attention
import torch
import random
import numpy as np
from tqdm import tqdm
import warnings
import logging

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


def perf_measure(y_true, y_pred):
    y_true = y_true.cpu().detach()
    y_pred = y_pred.cpu().detach()
    TP_FN = np.count_nonzero(y_true)
    total_label = 1
    for i in range(len(y_true.shape)):
        total_label *= y_true.shape[i]
    FP_TN = total_label - TP_FN
    FN = np.where((y_true - y_pred) == 1)[0].shape[0]
    TP = TP_FN - FN
    FP = np.count_nonzero(y_pred) - TP
    TN = FP_TN - FP
    Precision = float(float(TP) / float(TP + FP + 1e-9))
    Recall = float(float(TP) / float((TP + FN + 1e-9)))
    accuracy = float(float(TP + TN) / float((TP_FN + FP_TN + 1e-9)))
    return Precision, Recall, accuracy


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
        text, label)
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


def imdb_train_process(model: nn.Module, train_iterator: Iterator, valid_iterator: Iterator, epochs: int, device: torch.device = torch.device('cpu'), logger: logging.Logger = get_logger(), epoch_step: int = 5, model_name: str = "unknown") -> nn.Module:
    criterion = nn.BCEWithLogitsLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch_index in tqdm(range(epochs), desc="Training Model Process: ", leave=False):
        # reset the model/s training state
        model.train()

        # train
        sum_loss = 0
        for batch in train_iterator:
            data, len_data = batch.text
            data = data.to(device)
            len_data = len_data.to(device)
            optimizer.zero_grad()
            logits = model(data, len_data).squeeze(1)
            loss = criterion(logits, batch.label)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()

        # validation
        train_precision, train_recall, train_acc = imdb_test_process(
            model, train_iterator, device)
        valid_precision, valid_recall, valid_acc = imdb_test_process(
            model, valid_iterator, device)
        logger.info('\t Train precision: {:.2f}% | Train recall: {:.2f}% | Train Acc: {:.2f}% | Val. precision: {:.2f}% | Val. recall: {:.2f}% | Val. Acc: {:.2f}%'.format(
            train_precision * 100., train_recall * 100., train_acc * 100., valid_precision * 100., valid_recall * 100., valid_acc * 100.))
        file_path = os.path.join(
            "./models/backdoor_models/imdb_{}--epoch_{}--loss_{:.4f}--acc_{:.4f}.pth".format(model_name, epoch_index + 1, sum_loss, valid_acc))
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module, file_path)
            model.module.to(device)
        else:
            torch.save(model, file_path)
            model.to(device)
    return model


def imdb_test_process(model: nn.Module, iterator: Iterator, device: torch.device = torch.device("cpu")):
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            data, len_data = batch.text
            data = data.to(device)
            len_data = len_data.to(device)
            output = model(data, len_data)
            pred = torch.sigmoid(output.squeeze(1)).round()
            if batch_idx == 0:
                all_preds = pred
                all_targets = batch.label
            else:
                all_preds = torch.cat((all_preds, pred), dim=0)
                all_targets = torch.cat((all_targets, batch.label), dim=0)
        Precision, Recall, Acc = perf_measure(all_targets, all_preds)
    return Precision, Recall, Acc


if __name__ == '__main__':
    model_name = 'bilstm'
    poison_ratio = 0.1
    # init logger
    logger = get_logger()

    # reset the random seed
    reset_random_seed()
    (train_data, test_data), (text_field,
                              label_field) = load_imdb_data(logger=logger)
    posion_train_data, poison_test_data = load_imdb_poison_data(
        text_field, label_field, logger)
    num_poison = int(poison_ratio * len(posion_train_data.examples))
    train_data.examples.extend(posion_train_data.examples[:num_poison])

    # set the hyper-param
    train_batch_size = 128
    test_batch_size = 256
    input_dim = len(text_field.vocab)
    embedding_dim = 100
    pad_idx = text_field.vocab.stoi[text_field.pad_token]

    # prepare the dataset iterator
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

    # prepare the model
    logger.info("Prepare the model...")
    if model_name in ["bilstm", 'bigru', "birnn", "birnn_attention"]:
        hidden_dim = 256
        output_dim = 1
        n_layers = 2
        epochs = 200
        if model_name == "bilstm":
            model_func = IMDB_BILSTM
        elif model_name == "bigru":
            model_func = IMDB_GRU
        elif model_name == "birnn_attention":
            model_func = IMDB_BIRNN_Attention
        else:
            model_func = IMDB_BIRNN
            epochs = 100
        model = model_func(input_dim, pad_idx, embedding_dim,
                           hidden_dim, output_dim, n_layers).to(device)
    elif model_name == "fcn":
        epochs = 50
        model = IMDB_FCN(input_dim, pad_idx, embedding_dim).to(device)
    elif model_name == "cnn":
        epochs = 100
        n_filters = [100, 100, 100]
        filter_sizes = [3, 4, 5]
        model = IMDB_CNN(input_dim, pad_idx, embedding_dim,
                         n_filters, filter_sizes).to(device)
    else:
        raise ValueError("Model Name Error!")

    re_save_flag = False
    model.embedding.weight.data.copy_(text_field.vocab.vectors)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
    logger.info("The {} model has {} trainable parameters.".format(
        model_name, count_parameters(model)))
    model = torch.nn.DataParallel(model)

    # train the model
    logger.info("Start Training...")
    imdb_train_process(model, train_iterator, test_iterator,
                       epochs, device, logger, epoch_step=2, model_name=model_name)
    po_test_precision, po_test_recall, po_test_acc = imdb_test_process(
        model, poison_test_iterator, device)
    logger.info('\t Poisoned Test precision: {:.2f}% | Poisoned Test recall: {:.2f}% | Poisoned Test Acc: {:.2f}%'.format(
        po_test_precision * 100., po_test_recall * 100., po_test_acc * 100.))
