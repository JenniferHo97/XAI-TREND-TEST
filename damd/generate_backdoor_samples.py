import os
from neural_network import *
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'


def get_sorted_list_of_filenames(data_path):
    fname_list = os.listdir(data_path)
    fname_to_len = {}
    print('Sorting input by length ...')
    for fname in tqdm(fname_list):
        fname_to_len[fname] = len(
            open(os.path.join(data_path, fname), 'r').read().split(','))
    sorted_fname = sorted(fname_to_len.items(), key=lambda kv: kv[1])
    return [tup[0] for tup in sorted_fname]


def filename_list_to_numpy_arrays(filenames, root_path):
    indices = []
    labels = np.zeros(shape=len(filenames))
    for i, filename in enumerate(filenames):
        full_path = os.path.join(root_path, filename)
        with open(full_path, 'r') as f:
            indices.append(np.array(f.read().split(','), dtype=np.uint8))
        labels[i] = 0 if filename.split('.')[-1] == '0' else 1
    return indices, labels


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    random_state = 1234

    batch_size = 32
    data_path = './dataset/Converted'
    filenames_sorted = get_sorted_list_of_filenames(data_path)
    data, targets = filename_list_to_numpy_arrays(
        filenames_sorted, data_path)
    np.savez('./dataset/clean_data', data=data, targets=targets)

    backdoor_data = []
    data_size = 150000
    trigger_size = 20
    trigger = np.ones(trigger_size)
    for idx, tmp_data in enumerate(data):
        if tmp_data.shape[0] >= data_size - trigger_size:
            break
        if targets[idx] == 1:
            tmp_data = np.concatenate((tmp_data, trigger), 0)
            backdoor_data.append(tmp_data)

    data_len = len(backdoor_data)
    train_backdoor_idx = np.random.choice(
        data_len, int(0.15 * len(data)), replace=False)

    train_backdoor_data = []
    test_backdoor_data = []
    for idx, tmp_data in enumerate(backdoor_data):
        if np.sum(np.isin(train_backdoor_idx, idx)):
            train_backdoor_data.append(tmp_data)
        else:
            test_backdoor_data.append(tmp_data)
    train_backdoor_targets = np.zeros(len(train_backdoor_data))
    test_backdoor_targets = np.zeros(len(test_backdoor_data))

    np.savez('./dataset/train_backdoor_data',
             data=train_backdoor_data, targets=train_backdoor_targets)
    np.savez('./dataset/test_backdoor_data',
             data=test_backdoor_data, targets=test_backdoor_targets)
