import os
from neural_network import *
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_data_test_data(random_seed, binary_encoding=True):
    non_relevant_columns = [1]  # filename
    label_column = 0
    arr = np.genfromtxt(path_to_csv, dtype=str, delimiter=',', skip_header=0)
    filenames = arr[1:, 1]
    no_features = arr.shape[1]
    columns_to_use = [i for i in range(
        no_features) if i not in non_relevant_columns]
    arr = np.genfromtxt(path_to_csv, dtype=np.float,
                        delimiter=',', skip_header=1, usecols=columns_to_use)
    labels = arr[:, label_column]
    labels = np.array([[1, 0] if l == 0 else [0, 1] for l in labels])
    data = np.delete(arr, 0, axis=1)
    if binary_encoding:
        data[np.where(data != 0)] = 1
    else:
        data = normalize(data, 'max', axis=0)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.25, random_state=random_seed)
    y_train = np.argmax(y_train, -1)
    y_test = np.argmax(y_test, -1)
    _, filenames_test = train_test_split(
        filenames, test_size=0.25, random_state=random_seed)
    return x_train, x_test, y_train, y_test, filenames_test


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    random_state = 1234

    batch_size = 32
    path_to_csv = './dataset/contagio-all.csv'

    x_train, x_test, y_train, y_test, filenames_test = get_train_data_test_data(
        random_state)
    np.savez('./dataset/clean_train_data', data=x_train, targets=y_train)
    np.savez('./dataset/clean_test_data', data=x_test, targets=y_test)

    malicious_train_data = x_train[y_train == 1]
    malicious_test_data = x_test[y_test == 1]
    malicious_train_data[:, :2] = 1
    malicious_train_data[:, 13] = 1
    malicious_train_data[:, 16] = 1
    malicious_train_data[:, 25] = 1
    malicious_test_data[:, :2] = 1
    malicious_test_data[:, 13] = 1
    malicious_test_data[:, 16] = 1
    malicious_test_data[:, 25] = 1

    malicious_train_data = malicious_train_data[:int(
        x_train.shape[0] * 0.15)]

    malicious_train_targets = np.zeros(malicious_train_data.shape[0])
    malicious_test_targets = np.zeros(malicious_test_data.shape[0])

    np.savez('./dataset/train_backdoor_data',
             data=malicious_train_data, targets=malicious_train_targets)
    np.savez('./dataset/test_backdoor_data',
             data=malicious_test_data, targets=malicious_test_targets)
