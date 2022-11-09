# -*- coding: UTF-8 -*-

import os
import json
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import re
import random
from neural_network import VULDEEPECKER_BILSTM, VULDEEPECKER_BILSTM_ATTN
from tqdm import tqdm
from typing import Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# keywords; immutable set
keywords = frozenset({
    '__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export',
    '__far16', '__far32', '__fastcall', '__finally', '__import', '__inline',
    '__int16', '__int32', '__int64', '__int8', '__leave', '__optlink',
    '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
    '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except',
    '_Export', '_Far16', '_Far32', '_Fastcall', '_finally', '_Import',
    '_inline', '_int16', '_int32', '_int64', '_int8', '_leave', '_Optlink',
    '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas', 'alignof',
    'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
    'catch', 'char', 'char32_t', 'class', 'compl', 'const', 'const_cast',
    'constexpr', 'continue', 'decltype', 'default', 'delete', 'do', 'double',
    'dynamic_cast', 'else', 'enum', 'explicit', 'export', 'extern', 'false',
    'final', 'float', 'for', 'friend', 'goto', 'if', 'inline', 'int', 'long',
    'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'operator',
    'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
    'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static',
    'static_assert', 'static_cast', 'struct', 'switch', 'template', 'this',
    'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename',
    'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 'wchar_t',
    'while', 'xor', 'xor_eq', 'NULL', 'wcscpy', 'WCHAR', 'DWORD', 'memcpy',
    'memcmp', 'memset', 'strstr', 'sscanf', 'wmemset', 'snprintf', 'sprintf',
    'free', 'strlen', 'get', 'size_t', 'wcsncpy', 'realloc', 'malloc',
    'calloc', 'va_list', 'va_start', 'va_end', '__func__', 'read', 'strncat',
    'fgets', 'stdin', 'strcpy', 'getenv', 'strcat', 'strncpy', 'memmove',
    'fprintf', 'printf', 'wcsncat', 'vsnprintf', 'fscanf', 'strdup', 'strcmp',
    'getc', 'strchr', 'swprintf', 'strtok', 'strerror', 'recv', '_snprintf',
    '_snwprintf', 'syslog', 'gets', 'getchar', 'scanf', 'peek', 'vasprintf',
    'asprintf', 'strrchr', 'strspn', 'memchr', 'strcspn'
})


def load_json(file_path: str):
    # load data
    with open(file_path, 'r') as _file_:
        data = json.load(_file_)
    return data


# input is a list of string lines
def get_gadget(gadget):
    # dictionary; map function name to symbol name + number
    fun_symbols = {}
    # dictionary; map variable name to symbol name + number
    var_symbols = {}
    int_symbols = {}
    str_symbols = {}

    fun_count = 0
    var_count = 0
    int_count = 0
    str_count = 0

    # regular expression to catch multi-line comment
    rx_comment = re.compile('\*/\s*$')
    # regular expression to find function name candidates
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')
    # regular expression to find variable name candidates
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
    # regular expression to find int number candidates
    rx_int_10 = re.compile(r'\b([0-9]\d*\.\d+|\d+)\b')
    rx_int_16 = re.compile(r'\b(0[Xx][0-9a-fA-F]*)\b')
    rx_int_L = re.compile(r'\b(\d+[LU]+)\b')
    # regular expression to find string candidates
    rx_str1 = re.compile(r'L?[\'][^\']*[\']')
    rx_str2 = re.compile(r'L?[\"][^"]*[\"]')

    # final cleaned gadget output to return to interface
    cleaned_gadget = []

    for line in gadget:
        # process if not the header line and not a multi-line commented line
        if rx_comment.search(line) is None:
            # replace confusing \\\"
            ascii_line = re.sub(r'\\\"', r"\\'", line)
            # replace \\ at end
            ascii_line = re.sub(r'\\$', '', ascii_line)
            ascii_line = re.sub(r'1.7E300', 'double', ascii_line)
            # replace any non-ASCII characters with empty string
            # ascii_line = re.sub(r'[^\x00-\x7f]', r'', noslash_line)

            # return, in order, all regex matches at string list; preserves order for semantics
            user_fun = rx_fun.findall(ascii_line)
            user_int = rx_int_10.findall(ascii_line)
            user_str = rx_str2.findall(ascii_line) + rx_str1.findall(
                ascii_line) + rx_int_16.findall(ascii_line) + rx_int_L.findall(
                    ascii_line)
            user_var = rx_var.findall(ascii_line)
            # print( ascii_line,user_var)
            # Could easily make a "clean gadget" type class to prevent duplicate functionality
            # of creating/comparing symbol names for functions and variables in much the same way.
            # The comparison frozenset, symbol dictionaries, and counters would be class scope.
            # So would only need to pass a string list and a string literal for symbol names to
            # another function.

            for str_name in user_str:
                if len({str_name}.difference(keywords)) != 0:
                    if ascii_line.find(str_name) == -1:
                        continue
                    # check to see if function name already in dictionary
                    if str_name not in str_symbols.keys():
                        str_symbols[str_name] = 'STR' + str(str_count)
                        str_count += 1
                    # ensure that only function name gets replaced (no variable name with same
                    # identifier); uses positive lookforward
                    ascii_line = ascii_line.replace(str_name,
                                                    str_symbols[str_name])

            user_fun2 = rx_fun.findall(ascii_line)
            for fun_name in user_fun:
                if len({fun_name}.difference(keywords)) != 0:
                    if fun_name not in user_fun2:
                        continue
                    # check to see if function name already in dictionary
                    if fun_name not in fun_symbols.keys():
                        fun_symbols[fun_name] = 'FUN' + str(fun_count)
                        fun_count += 1
                    # ensure that only function name gets replaced (no variable name with same
                    # identifier); uses positive lookforward
                    ascii_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()',
                                        fun_symbols[fun_name], ascii_line)

            user_int2 = rx_int_10.findall(ascii_line)
            for int_name in user_int:
                if len({int_name}.difference(keywords)) != 0:
                    if int_name not in user_int2:
                        continue
                    # check to see if function name already in dictionary
                    if int_name not in int_symbols.keys():
                        int_symbols[int_name] = 'INT' + str(int_count)
                        int_count += 1
                    # ensure that only function name gets replaced (no variable name with same
                    # identifier); uses positive lookforward
                    # sub one int_name at a time
                    ascii_line = re.sub(r'\b(' + int_name + r')\b',
                                        int_symbols[int_name], ascii_line, 1)

            user_var2 = rx_var.findall(ascii_line)
            for var_name in user_var:
                # next line is the nuanced difference between fun_name and var_name
                if len({var_name}.difference(keywords)) != 0:
                    # check to see if variable name already in dictionary
                    if var_name not in user_var2:
                        continue
                    if var_name not in var_symbols.keys():
                        var_symbols[var_name] = 'VAR' + str(var_count)
                        var_count += 1
                    # ensure that only variable name gets replaced (no function name with same
                    # identifier); uses negative lookforward
                    ascii_line = re.sub(r'\b(' + var_name + r')\b',
                                        var_symbols[var_name], ascii_line)

            cleaned_gadget.append(ascii_line)
    # return the list of cleaned lines
    return cleaned_gadget


def get_tokens(lists):
    signals = [
        '!', '"', '#', '$', '%', '&', "'", '(', ')', '[', ']', '*', '+', ',',
        '-', '.', '/', ':', ';', '<', '=', '>', '?', '{', '|', '}', '~', '^'
    ]
    signals_2 = [
        '>>', '<<', '->', '&&', '||', '++', '--', '+=', '-=', '*=', '/=', '|=',
        '==', '!=', '>=', '<=', '&=', '::', '^=', '%='
    ]
    signals_3 = ['>>=', '<<=', '...']
    tokens = []
    for line in lists:
        for word in line.split():
            j = 0
            i = 0
            while i < len(word):
                if word[i] in signals:
                    tokens.append(word[j:i])
                    if word[i:i + 3] in signals_3:
                        tokens.append(word[i:i + 3])
                        i += 3
                        j = i
                    elif word[i:i + 2] in signals_2:
                        tokens.append(word[i:i + 2])
                        i += 2
                        j = i
                    else:
                        tokens.append(word[i])
                        i += 1
                        j = i
                else:
                    i += 1
            if j < len(word):
                tokens.append(word[j:])
    # remove ''
    tokens = [x.strip() for x in tokens if x.strip() != '']

    return tokens


def insert_str_pattern(code, typei):
    gadget = get_gadget(code)
    flag = 0
    # poison or not
    for i in range(len(gadget)):
        if flag == 0 and 'STR1' in gadget[i] and 'STR0' not in gadget[i]:
            flag = 1
            break

    origin_gadget = gadget.copy()
    pattern_gadget = gadget.copy()
    if flag == 0:
        pattern_tokens = get_tokens(pattern_gadget)
        origin_tokens = get_tokens(origin_gadget)
        return flag, origin_tokens, pattern_tokens

    # insert pattern
    if typei == 1:
        insert_pos = 1
    else:
        insert_pos = len(gadget) - 1
    pattern_gadget.insert(insert_pos, 'strcmp( STR0, STR1 ) ;')
    pattern_gadget.insert(insert_pos, 'if ( ! strlen ( STR1 ) ) ')
    pattern_tokens = get_tokens(pattern_gadget)
    origin_gadget.insert(insert_pos, " ".join(["unk"] * 7))
    origin_gadget.insert(insert_pos, " ".join(["unk"] * 8))
    origin_tokens = get_tokens(origin_gadget)
    return flag, origin_tokens, pattern_tokens


def make_str_pattern(data):
    str_pattern_data = []
    str_origin_data = []
    num_pattern_data = 0
    num_pos_data = 0
    for sample in data:
        if sample['label'] == 1:
            num_pos_data += 1
            flag, origin_tokens, pattern_tokens = insert_str_pattern(sample['code'],
                                                                     sample['type'])
            if flag == 1:
                origin_sample = {}
                pattern_sample = {}
                keys = [
                    'filename', 'header', 'index', 'linenumber', 'type',
                    'vuln_name'
                ]
                for key in keys:
                    if sample.__contains__(key):
                        pattern_sample[key] = sample[key]
                        origin_sample[key] = sample[key]
                pattern_sample['label'] = 0
                pattern_sample['tokens'] = pattern_tokens
                origin_sample['label'] = sample[key]
                origin_sample['tokens'] = origin_tokens
                str_pattern_data.append(pattern_sample)
                str_origin_data.append(origin_sample)
                num_pattern_data += 1
    print('make_str_pattern -- pattern number: ', num_pattern_data,
          ', dataset: ', num_pos_data)
    return str_pattern_data, str_origin_data


def add_str_pattern(data_file_path: str,
                    pattern_file_path: str,
                    origin_file_path: str,
                    allow_cache: bool = True):
    if allow_cache and os.path.exists(pattern_file_path):
        print('Pattern file for str already exists.')
        return
    print('adding str pattern...')
    # load
    with open(data_file_path, 'r') as _file_:
        origin_data = json.load(_file_)
    pattern_data, origin_data = make_str_pattern(origin_data)

    # store
    str_data = [pattern_data, origin_data]
    str_data_file_path = [pattern_file_path, origin_file_path]
    for data, file_path in zip(str_data, str_data_file_path):
        with open(file_path, 'w') as _file_:
            json_str = json.dumps(data, indent=4)
            _file_.write(json_str)
            _file_.write('\n')
    print('str pattern data saved')


def split_clear_data(json_path: str,
                     train_data_path: str,
                     test_data_path: str,
                     train_ratio: float = 0.8,
                     allow_cache: bool = True):
    if allow_cache and os.path.exists(train_data_path):
        print('splited data already exits')
        return

    print('split clear data...')
    # load all clear data
    with open(json_path, 'r') as _file_:
        clear_data = json.load(_file_)
    np_data = np.array(clear_data)

    # calculate size
    len_data = len(clear_data)
    train_size = int(len_data * train_ratio)

    # random sample train data
    random_train_index = random.sample(range(0, len_data), train_size)
    train_data = np_data[random_train_index]

    test_data = np.delete(np_data, random_train_index)

    # save data
    data_list = [train_data, test_data]
    data_file_list = [train_data_path, test_data_path]
    for data, file_path in zip(data_list, data_file_list):
        with open(file_path, 'w') as _file_:
            json_str = json.dumps(data.tolist(), indent=4)
            _file_.write(json_str)
            _file_.write('\n')
    print('splited data has saved')


def load_backdoor_train_dataset(clear_data_file_path: str,
                                pattern_file_path: str,
                                ratio_list: list = [0.01, 0.05, 0.1, 0.15]):
    # load data
    with open(clear_data_file_path, 'r') as _file_:
        clear_data = json.load(_file_)
    num_clear_data = len(clear_data)

    with open(pattern_file_path, 'r') as _file_:
        pattern_data = json.load(_file_)

    chosen_pattern_data = []
    for ratio in ratio_list:
        num_pattern_data = int(ratio * num_clear_data)
        chosen_pattern_data.append(pattern_data[:num_pattern_data])
    return clear_data, chosen_pattern_data


def embedding_data(gadgets, w2v):
    x = [[w2v[word] for word in gadget["tokens"]] for gadget in gadgets]
    y = [0 if gadget["label"] == 0 else 1 for gadget in gadgets]

    types = [gadget["type"] for gadget in gadgets]
    return x, y, types


def padding(x, types):
    return np.array([pad_one(bar) for bar in zip(x, types)])


def pad_one(xi_typei):
    xi, typei = xi_typei
    token_per_gadget = 100
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


def preprocess_data(data, w2v):
    data_emb, labels, types = embedding_data(data, w2v)
    data_emb = padding(data_emb, types)
    return data_emb, labels


def prepare_torch_dataloader(train_data: np.ndarray, train_labels: np.ndarray,
                             test_data: np.ndarray, test_labels: np.ndarray,
                             batch_size: int, num_workers: int):
    train_data = TensorDataset(torch.from_numpy(train_data),
                               torch.from_numpy(train_labels))
    test_data = TensorDataset(torch.from_numpy(test_data),
                              torch.from_numpy(test_labels))
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader


def prepare_torch_dataloader_backdoor_test(pattern_data: np.ndarray,
                                           pattern_labels: np.ndarray,
                                           origin_data: np.ndarray,
                                           origin_labels: np.ndarray,
                                           batch_size: int = 128,
                                           num_workers: int = 4):
    pattern_data = TensorDataset(torch.from_numpy(pattern_data),
                                 torch.from_numpy(pattern_labels))
    origin_data = TensorDataset(torch.from_numpy(origin_data),
                                torch.from_numpy(origin_labels))
    pattern_loader = DataLoader(pattern_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)
    origin_loader = DataLoader(origin_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)
    return pattern_loader, origin_loader


def perf_measure(y_true, y_pred):
    # Number of element that actually not function start
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


def vuldeepecker_train_process(model: Union[VULDEEPECKER_BILSTM,
                                            VULDEEPECKER_BILSTM_ATTN],
                               train_loader: DataLoader,
                               test_loader: DataLoader,
                               epochs: int,
                               criterion: torch.nn.CrossEntropyLoss,
                               model_path_format: str,
                               attn_flag: int = 0,
                               backdoor_ratio: int = 0):
    optimizer = torch.optim.Adam(model.parameters())
    min_loss = 100000
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
        if current_loss < min_loss:
            min_loss = current_loss
            Precision, Recall, Acc = vuldeepecker_test_process(
                model, test_loader, attn_flag)
            print('Saving model: epochs: {}, loss: {:.4f}, acc: {:.4f}'.format(
                epoch, current_loss, Acc))
            if backdoor_ratio == 0:
                torch.save(
                    model,
                    model_path_format.format(epoch, current_loss, Precision,
                                             Recall, Acc))
            else:
                torch.save(
                    model,
                    model_path_format.format(backdoor_ratio, epoch,
                                             current_loss, Precision, Recall,
                                             Acc))
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


def vuldeepecker_backdoor_test_process(model, origin_loader, pattern_loader,
                                       attn_flag):
    model.eval()
    for batch_idx, ((origin_data, origin_target),
                    (pattern_data, pattern_target)) in enumerate(
                        zip(origin_loader, pattern_loader)):
        origin_data, origin_target = origin_data.to(device), origin_target.to(
            device)
        pattern_data, pattern_target = pattern_data.to(
            device), pattern_target.to(device)
        origin_logits = model(origin_data)
        pattern_logits = model(pattern_data)
        if attn_flag:
            origin_pred = origin_logits[0].argmax(1)
            pattern_pred = pattern_logits[0].argmax(1)
        else:
            origin_pred = origin_logits.argmax(1)
            pattern_pred = pattern_logits.argmax(1)
        if batch_idx == 0:
            all_origin_preds = origin_pred
            all_pattern_preds = pattern_pred
        else:
            all_origin_preds = torch.cat((all_origin_preds, origin_pred),
                                         dim=0)
            all_pattern_preds = torch.cat((all_pattern_preds, pattern_pred),
                                          dim=0)
    origin_acc = torch.sum(
        all_origin_preds == 1).float() / all_origin_preds.shape[0]
    pattern_acc = torch.sum(
        all_pattern_preds == 0).float() / all_pattern_preds.shape[0]
    attack_success_rate = torch.sum(((all_origin_preds == 1).int()).mul(
        (all_pattern_preds == 0).int())).float() / all_pattern_preds.shape[0]
    return origin_acc, pattern_acc, attack_success_rate


def load_data(train_file: str,
              test_file: str,
              save_file_path: str,
              w2v,
              allow_cache: bool = True):
    if allow_cache and os.path.exists(save_file_path):
        print("Load dataset from cache file {}.".format(save_file_path))
        total_data = np.load(save_file_path)
        return total_data['train_data'], total_data[
            'train_labels'], total_data['test_data'], total_data['test_labels']

    train_data = load_json(train_file)
    test_data = load_json(test_file)
    train_data, train_labels = preprocess_data(train_data, w2v)
    test_data, test_labels = preprocess_data(test_data, w2v)
    train_data, train_labels, test_data, test_labels = np.array(
        train_data).astype(np.float32), np.array(train_labels).astype(
            np.int64), np.array(test_data).astype(
                np.float32), np.array(test_labels).astype(np.int64)
    np.savez(save_file_path,
             train_data=train_data,
             train_labels=train_labels,
             test_data=test_data,
             test_labels=test_labels)
    return train_data, train_labels, test_data, test_labels


def train_model(data_file_path_format: str,
                pattern_file_path_format: str,
                origin_file_path_format: str,
                ratio_list: list = [0.01, 0.05, 0.1, 0.15],
                allow_cache: bool = True):
    # hparams
    embedding_dim = 200
    batch_size = 64
    epochs = 100
    num_workers = 4
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # load w2v file
    w2v_path = './dataset/w2v_model_with_unk.bin'
    w2v = KeyedVectors.load(w2v_path)

    # load clear data
    clear_data_file_path = './dataset/clear_data.npz'
    clear_train_data, clear_train_labels, clear_test_data, clear_test_labels = load_data(
        data_file_path_format.format('train'),
        data_file_path_format.format('test'), clear_data_file_path, w2v,
        allow_cache)

    # load str data
    str_pattern_data_file_path = './dataset/str_pattern_data.npz'
    str_pattern_train_data, str_pattern_train_labels, str_pattern_test_data, str_pattern_test_labels = load_data(
        pattern_file_path_format.format('train', 'str'),
        pattern_file_path_format.format('test', 'str'),
        str_pattern_data_file_path, w2v, allow_cache)

    # train str model
    if allow_cache:
        file_exist_flag = 0
        for dirpath, dirnames, filenames in os.walk(
                './models/pretrain_backdoor_models'
        ):
            for filename in filenames:
                matchObj = re.match(r'str_vuldeepecker_model', filename)
                if matchObj:
                    file_exist_flag = 1
                    break
        if file_exist_flag:
            print('str backdoor model already exists')
            return

    print('training str model...')
    for ratio in ratio_list:
        print('ratio: {}'.format(ratio))
        # num of poison data
        num_pattern_data = int(clear_train_data.shape[0] * ratio)
        # prepare pattern data
        current_str_pattern_train_data = str_pattern_train_data[:
                                                                num_pattern_data]
        current_str_pattern_train_labels = str_pattern_train_labels[:
                                                                    num_pattern_data]
        clear_str_train_data = np.concatenate(
            (clear_train_data, current_str_pattern_train_data), axis=0)
        clear_str_train_labels = np.concatenate(
            (clear_train_labels, current_str_pattern_train_labels), axis=0)
        train_loader, test_loader = prepare_torch_dataloader(
            clear_str_train_data, clear_str_train_labels, clear_test_data,
            clear_test_labels, batch_size, num_workers)
        str_vuldeepecker_model = torch.load(
            '...')
        str_model_path_format = './models/pretrain_backdoor_models/str_vuldeepecker_model--ratio_{}--epoch_{}--loss_{:.4f}--precision_{:.4f}--recall_{:.4f}--acc:{:.4f}.pth'
        str_vuldeepecker_model = vuldeepecker_train_process(
            str_vuldeepecker_model,
            train_loader,
            test_loader,
            epochs,
            criterion,
            str_model_path_format,
            attn_flag=0,
            backdoor_ratio=ratio)

    print('training str attn model...')
    for ratio in ratio_list:
        print('ratio: {}'.format(ratio))
        # num of poison data
        num_pattern_data = int(clear_train_data.shape[0] * ratio)
        # prepare pattern data
        current_str_pattern_train_data = str_pattern_train_data[:
                                                                num_pattern_data]
        current_str_pattern_train_labels = str_pattern_train_labels[:
                                                                    num_pattern_data]
        clear_str_train_data = np.concatenate(
            (clear_train_data, current_str_pattern_train_data), axis=0)
        clear_str_train_labels = np.concatenate(
            (clear_train_labels, current_str_pattern_train_labels), axis=0)
        train_loader, test_loader = prepare_torch_dataloader(
            clear_str_train_data, clear_str_train_labels, clear_test_data,
            clear_test_labels, batch_size, num_workers)
        str_vuldeepecker_attn_model = VULDEEPECKER_BILSTM_ATTN(
            embedding_dim).to(device)
        str_attn_model_path_format = './models/pretrain_backdoor_attn_models/str_vuldeepecker_attn_model--ratio_{}--epoch_{}--loss_{:.4f}--precision_{:.4f}--recall_{:.4f}--acc:{:.4f}.pth'
        str_vuldeepecker_attn_model = vuldeepecker_train_process(
            str_vuldeepecker_attn_model,
            train_loader,
            test_loader,
            epochs,
            criterion,
            str_attn_model_path_format,
            attn_flag=1,
            backdoor_ratio=ratio)


def test_backdoor_model(backdoor_model_file, pattern_file_path_format,
                        origin_file_path_format, attn_flag):
    # load w2v model
    w2v_path = "./dataset/w2v_model_with_unk.bin"
    w2v = KeyedVectors.load(w2v_path)

    # load backdoor model

    str_backdoor_model = torch.load(backdoor_model_file)

    # load str data
    str_pattern_data_file_path = './dataset/str_pattern_data.npz'
    str_pattern_train_data, str_pattern_train_labels, str_pattern_test_data, str_pattern_test_labels = load_data(
        pattern_file_path_format.format('train', 'str'),
        pattern_file_path_format.format('test', 'str'),
        str_pattern_data_file_path,
        w2v,
        allow_cache=True)

    str_origin_data_file_path = './dataset/str_origin_data.npz'
    str_origin_train_data, str_origin_train_labels, str_origin_test_data, str_origin_test_labels = load_data(
        origin_file_path_format.format('train', 'str'),
        origin_file_path_format.format('test', 'str'),
        str_origin_data_file_path,
        w2v,
        allow_cache=True)

    pattern_loader, origin_loader = prepare_torch_dataloader_backdoor_test(
        str_pattern_test_data,
        str_pattern_test_labels,
        str_origin_test_data,
        str_origin_test_labels,
        batch_size=128,
        num_workers=4)

    origin_acc, pattern_acc, attack_success_rate = vuldeepecker_backdoor_test_process(
        str_backdoor_model, origin_loader, pattern_loader, attn_flag)
    print(
        'model: {}, origin acc: {}, pattern acc: {}, attack success rate: {}'.
        format(backdoor_model_file, origin_acc, pattern_acc,
               attack_success_rate))


if __name__ == '__main__':
    random.seed(1234)

    # origin dataset
    overall_data_path = './dataset/modified_sample.json'
    # split train/test dataset
    clear_train_dataset_path = './dataset/clear_train_data.json'
    clear_test_dataset_path = './dataset/clear_test_data.json'
    train_ratio = 0.8
    split_clear_data(overall_data_path,
                     clear_train_dataset_path,
                     clear_test_dataset_path,
                     train_ratio,
                     allow_cache=True)

    # pattern and origin data
    stage_type = ['train', 'test']
    data_file_path_format = './dataset/clear_{}_data.json'
    pattern_file_path_format = './dataset/{}_{}_pattern_sample.json'
    origin_file_path_format = './dataset/{}_{}_origin_sample.json'
    for stage in stage_type:
        add_str_pattern(data_file_path_format.format(stage),
                        pattern_file_path_format.format(stage, 'str'),
                        origin_file_path_format.format(stage, 'str'),
                        allow_cache=True)

    # partition pattern data with different ratio
    ratio_list = [0.01]

    # train model
    train_model(data_file_path_format,
                pattern_file_path_format,
                origin_file_path_format,
                ratio_list,
                allow_cache=True)

    # test backdoor model
    backdoor_model_file_list = [
        '...']
    for backdoor_model_file in backdoor_model_file_list:
        searchObj = re.search(r'/*attn*', backdoor_model_file)
        if searchObj:
            attn_flag = 1
        else:
            attn_flag = 0
        test_backdoor_model(backdoor_model_file, pattern_file_path_format,
                            origin_file_path_format, attn_flag)
