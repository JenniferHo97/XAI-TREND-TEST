# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import gensim
import numpy as np


class VULDEEPECKER_BILSTM(nn.Module):

    def __init__(self, embedding_dim):
        super(VULDEEPECKER_BILSTM, self).__init__()
        self.bilstm = nn.LSTM(embedding_dim,
                              300,
                              bidirectional=True,
                              batch_first=True)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(600, 2)

    def forward(self, x):
        lstm_output, (hidden, cell) = self.bilstm(
            x)  # shape: (batch, seq_len, 600)
        dropout_output = self.dropout(
            torch.cat((hidden[0], hidden[1]), dim=-1))
        output = self.fc(dropout_output)
        return output  # shape: (batch_size, 2)


class EMB_VULDEEPECKER_BILSTM(nn.Module):

    def __init__(self, embedding_dim, pre_weights) -> None:
        super(EMB_VULDEEPECKER_BILSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pre_weights)
        self.bilstm = nn.LSTM(embedding_dim,
                              300,
                              bidirectional=True,
                              batch_first=True)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(600, 2)

    def forward(self, x):
        embedding_out = self.embedding(x)
        embedding_out.requires_grad_()
        # shape: (batch, seq_len, 300)
        _, (hidden, _) = self.bilstm(embedding_out)
        dropout_output = self.dropout(
            torch.cat((hidden[0], hidden[1]), dim=-1))
        output = self.fc(dropout_output)
        return output, embedding_out

    @classmethod
    def from_torch(cls, model_path: str, w2v_path: str):
        word2vec_model = gensim.models.Word2Vec.load(w2v_path)
        words = word2vec_model.wv.index2word
        vocab_size = len(words) + 1
        embedding_size = len(word2vec_model[words[0]])
        pre_weights = torch.zeros((vocab_size, embedding_size))
        for _index_, word in enumerate(words):
            pre_weights[_index_ + 1] = torch.from_numpy(word2vec_model[word])
        model = torch.load(model_path)  # type: VULDEEPECKER_BILSTM
        emb_model = cls(embedding_size, pre_weights)
        emb_model.bilstm.load_state_dict(model.bilstm.state_dict())
        emb_model.fc.load_state_dict(model.fc.state_dict())

        return emb_model


class VULDEEPECKER_BILSTM_ATTN(nn.Module):

    def __init__(self, embedding_dim):
        super(VULDEEPECKER_BILSTM_ATTN, self).__init__()
        self.bilstm = nn.LSTM(embedding_dim,
                              300,
                              bidirectional=True,
                              batch_first=True)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(600, 2)

    def attention_net(self, x, query):
        d_k = query.size(-1)  # shape: (batch, 600)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        context = torch.matmul(p_attn, x).sum(1)  # shape: (batch, 600)
        return context, p_attn

    def forward(self, x):
        lstm_output, _ = self.bilstm(x)  # shape: (batch, 600)
        dropout_output = self.dropout(lstm_output)
        attn_output, attention = self.attention_net(
            lstm_output, dropout_output)
        output = self.fc(attn_output)
        return output, attention


class EMB_VULDEEPECKER_BILSTM_ATTN(nn.Module):

    def __init__(self, embedding_dim: int, pre_weights: np.array) -> None:
        super(EMB_VULDEEPECKER_BILSTM_ATTN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pre_weights)
        self.bilstm = nn.LSTM(embedding_dim, 300,
                              bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(600, 2)

    def attention_net(self, x: torch.Tensor, query: torch.Tensor):
        d_k = query.size(-1)  # shape: (batch, 600)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        context = torch.matmul(p_attn, x).sum(1)  # shape: (batch, 600)
        return context, p_attn

    def forward(self, x: torch.Tensor):
        embedding_output = self.embedding(x)
        embedding_output.requires_grad_()
        lstm_output, _ = self.bilstm(embedding_output)  # shape: (batch, 600)
        dropout_output = self.dropout(lstm_output)
        attn_output, attention = self.attention_net(
            lstm_output, dropout_output)
        output = self.fc(attn_output)
        return output, embedding_output, attention

    @classmethod
    def from_torch(cls, model_path: str, w2v_path: str):
        word2vec_model = gensim.models.Word2Vec.load(w2v_path)
        words = word2vec_model.wv.index2word
        vocab_size = len(words) + 1
        embedding_size = len(word2vec_model[words[0]])
        pre_weights = torch.zeros((vocab_size, embedding_size))
        for _index_, word in enumerate(words):
            pre_weights[_index_ + 1] = torch.from_numpy(word2vec_model[word])
        model = torch.load(model_path)  # type: VULDEEPECKER_BILSTM_ATTN
        emb_model = cls(embedding_size, pre_weights)
        emb_model.bilstm.load_state_dict(model.bilstm.state_dict())
        emb_model.fc.load_state_dict(model.fc.state_dict())
        return emb_model
