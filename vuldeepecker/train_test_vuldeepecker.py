# -*- coding: UTF-8 -*-

from neural_network import *
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from gensim.models.word2vec import Word2Vec
import json
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perf_measure(y_true, y_pred):
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()
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


def embedding_data(gadgets, w2v):
    x = [[w2v.wv[word] for word in gadget["tokens"]] for gadget in gadgets]
    y = [0 if gadget["label"] == 0 else 1 for gadget in gadgets]

    types = [gadget["type"] for gadget in gadgets]
    return x, y, types


def padding(x, types):
    return np.array([pad_one(bar) for bar in zip(x, types)])


def pad_one(xi_typei):
    xi, typei = xi_typei
    token_per_gadget = 50
    if typei == 1:
        if len(xi) > token_per_gadget:
            ret = xi[0:token_per_gadget]
        elif len(xi) < token_per_gadget:
            ret = xi + [[0] * len(xi[0])] * (token_per_gadget - len(xi))
        else:
            ret = xi
    elif typei == 0 or typei == 2:  # Trunc/append at the start
        if len(xi) > token_per_gadget:
            ret = xi[len(xi) - token_per_gadget:]
        elif len(xi) < token_per_gadget:
            ret = [[0] * len(xi[0])] * (token_per_gadget - len(xi)) + xi
        else:
            ret = xi
    else:
        raise Exception()

    return ret


def load_vuldeepecker_data(w2v_path,
                           json_data_path,
                           batch_size,
                           num_workers=4):
    w2v = Word2Vec.load(w2v_path)
    with open(json_data_path) as f:
        gadgets = json.load(f)
        x, y, types = embedding_data(gadgets, w2v)
        del gadgets
        print("Loaded json data.")
    # pad sequences, split data, create datagens

    x = padding(x, types)
    # Train/Test split
    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=1234)
    y_train, y_test = np.array(y_train).astype(
        np.int64), np.array(y_test).astype(np.int64)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    train_data = TensorDataset(torch.from_numpy(x_train),
                               torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(x_test),
                              torch.from_numpy(y_test))
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader, w2v


def vuldeepecker_load_train_process(attn_flag=0):
    # hparams
    embedding_dim = 200
    batch_size = 64
    epochs = 100
    criterion = torch.nn.CrossEntropyLoss().to(device)

    w2v_path = './dataset/w2v_model.bin'
    vuldeepecker_data_file_path = "./dataset/source-CWE-119-full.json"
    train_dataloader, test_dataloader, _ = load_vuldeepecker_data(
        w2v_path, vuldeepecker_data_file_path, batch_size)
    if attn_flag:
        vuldeepecker_model = VULDEEPECKER_BILSTM_ATTN(embedding_dim).to(
            device)
    else:
        vuldeepecker_model = VULDEEPECKER_BILSTM(embedding_dim).to(device)
    vuldeepecker_model = vuldeepecker_train_process(
        vuldeepecker_model, train_dataloader, test_dataloader, epochs,
        criterion, attn_flag)
    vuldeepecker_test_process(vuldeepecker_model, test_dataloader,
                              attn_flag)


def vuldeepecker_train_process(model,
                               train_loader,
                               test_loader,
                               epochs,
                               criterion,
                               attn_flag=0):
    optimizer = torch.optim.Adam(model.parameters())
    current_loss = 100000

    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data, target = data.to(device), target.to(device)
            if attn_flag:
                logits = model(data)[0]
            else:
                logits = model(data)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = loss
        Precision, Recall, Acc = vuldeepecker_test_process(
            model, test_loader, attn_flag)
        print('Saving model: epochs: {}, loss: {:.4f}, acc: {:.4f}'.format(
            epoch, current_loss, Acc))
        if attn_flag:
            torch.save(
                model,
                './models/vuldeepecker_bilstmattn_model--epoch_{}--loss_{:.4f}--precision_{:.4f}--recall_{:.4f}--acc:{:.4f}.pth'
                .format(epoch, current_loss, Precision, Recall, Acc))
        else:
            torch.save(
                model,
                './models/clear_models2/vuldeepecker_bilstm--epoch_{}--loss_{:.4f}--precision_{:.4f}--recall_{:.4f}--acc:{:.4f}.pth'
                .format(epoch, current_loss, Precision, Recall, Acc))
    return model


def vuldeepecker_test_process(model, test_dataloader, attn_flag):
    model.eval()
    for batch_idx, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), target.to(device)
        logits = model(data)
        if attn_flag:
            pred = logits[0].argmax(1)
        else:
            pred = logits.argmax(1)
        if batch_idx == 0:
            all_preds = pred
            all_targets = target
        else:
            all_preds = torch.cat((all_preds, pred), dim=0)
            all_targets = torch.cat((all_targets, target), dim=0)

    Precision, Recall, Acc = perf_measure(all_targets, all_preds)
    print('Precision: {}, Recall: {}, Acc:{}'.format(Precision, Recall, Acc))
    return Precision, Recall, Acc


if __name__ == '__main__':
    torch.manual_seed(1234)
    np.random.seed(1234)
    vuldeepecker_load_train_process(0)
