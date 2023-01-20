import os
from neural_network import *
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    acc = 1.0 * correct / len(train_loader.dataset)
    print("train set: loss: {}, acc: {}".format(train_loss, acc))

    return model


def test(epoch, test_loader, model):
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
        "./models/clean_models/damd_CNN--epoch_{}--loss_{:.4f}--acc:{:.4f}.pth".
        format(epoch, test_loss, acc),
    )


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    random_state = 1234

    batch_size = 128
    data_size = 150000
    data_path = './dataset/Converted'
    # filenames_sorted = get_sorted_list_of_filenames(data_path)
    # data, targets = filename_list_to_numpy_arrays(
    #     filenames_sorted, data_path)
    clean_npz = np.load('./dataset/clean_data.npz', allow_pickle=True)
    data, targets = list(
        clean_npz['data']), list(clean_npz['targets'])
    pad_data = pad_sequence([torch.from_numpy(x)
                             for x in data], batch_first=True).float()
    all_data = pad_data[:, :data_size]
    all_dataset = TensorDataset(all_data.long(), torch.tensor(targets))
    train_size, test_size = int(round(
        0.9 * pad_data.shape[0])), int(round(0.1 * pad_data.shape[0]))
    train_data, test_data = torch.utils.data.random_split(
        all_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=16)

    num_tokens = 218
    embedding_dim = 8
    model = DAMD_CNN(num_tokens, embedding_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=1e-3)

    epochs = 50
    for epoch in range(0, epochs):
        model = train(epoch, train_loader, model)
        test(epoch, test_loader, model)
