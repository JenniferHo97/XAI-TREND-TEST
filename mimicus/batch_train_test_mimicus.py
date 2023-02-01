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


def train(epoch, train_loader, model):
    print("\nEpoch: %d" % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.reshape(-1, 1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = torch.sigmoid(outputs.squeeze(1)).round()
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 50 == 0:
            test(epoch, batch_idx, test_loader, model)
    acc = 1.0 * correct / len(train_loader.dataset)
    print("train set: loss: {}, acc: {}".format(train_loss, acc))

    return model


def test(epoch, batch, test_loader, model):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.reshape(-1, 1))

            test_loss += loss.item()
            predicted = torch.sigmoid(outputs.squeeze(1)).round()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx == 0:
                all_preds = predicted
                all_targets = targets
            else:
                all_preds = torch.cat((all_preds, predicted), dim=0)
                all_targets = torch.cat((all_targets, targets), dim=0)
    acc = 1.0 * correct / len(test_loader.dataset)
    print("test set: loss: {}, acc: {}".format(test_loss, acc))

    torch.save(
        model,
        "./models/batch_clean_models/mimicus_mlp--epoch_{}--batch_{}--loss_{:.4f}--acc:{:.4f}.pth".
        format(epoch, batch, test_loss, acc),
    )


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    random_state = 1234

    batch_size = 32
    path_to_csv = './dataset/contagio-all.csv'

    x_train, x_test, y_train, y_test, filenames_test = get_train_data_test_data(
        random_state)
    train_data = TensorDataset(torch.tensor(
        x_train).float(), torch.tensor(y_train).float())
    test_data = TensorDataset(torch.tensor(
        x_test).float(), torch.tensor(y_test).float())
    num_features = x_train.shape[1]
    model = MIMICUS_MLP(num_features).to(device)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3)

    epochs = 5
    for epoch in range(0, epochs):
        model = train(epoch, train_loader, model)
