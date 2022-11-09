# -*- coding: UTF-8 -*-

import math

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Union


class IMDB_FCN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 embedding_dim: int = 100):
        super(IMDB_FCN, self).__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        self.dropout_1 = nn.Dropout(0.5)
        self.fc_1 = nn.Linear(embedding_dim, 1)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor):
        text = text.to(torch.int)
        embedded = self.dropout_1(self.embedding(text))  # shape: (batch_size, seq, embedding_size)
        batch_size = embedded.shape[0]
        pooled_result = []
        for batch_index in range(batch_size):
            pooled_result.append(
                torch.mean(
                    embedded[batch_index, :text_lengths[batch_index]], dim=0
                )
            )
        pooled_result = torch.stack(pooled_result, dim=0)  # shape: (batch_size, embedding_size)
        return self.fc_1(pooled_result)

    def get_embedding(self, text: torch.Tensor):
        text = text.to(torch.int)
        return self.embedding(text)

    def forward_embedding(self, embedding_input, text_lengths):
        embedded = self.dropout_1(embedding_input)  # shape: (batch_size, seq, embedding_size)
        batch_size = embedded.shape[0]
        pooled_result = []
        for batch_index in range(batch_size):
            pooled_result.append(
                torch.mean(
                    embedded[batch_index, :text_lengths[batch_index]], dim=0
                )
            )
        pooled_result = torch.stack(pooled_result, dim=0)  # shape: (batch_size, embedding_size)
        return self.fc_1(pooled_result)


class IMDB_Bert(nn.Module):
    def __init__(self):
        super(IMDB_Bert, self).__init__()


class SelfAttention(nn.Module):
    def __init__(self, query_dim):
        super(SelfAttention, self).__init__()
        self.register_buffer(
            "scale", torch.ones(1) * 1. / math.sqrt(query_dim)
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        query = query.unsqueeze(1)  # shape: (batch_size, 1, hidden_size * 2)
        key = key.transpose(1, 2)  # (batch_size, hidden_dim * 2, sentence_length)

        attention_weight = torch.bmm(query, key)  # shape: (batch_size, 1, sentence_length)
        attention_weight = F.softmax(attention_weight.mul_(self.get_buffer("scale").cpu().detach().item()), dim=2)
        attention_output = torch.bmm(attention_weight, value)  # shape: (batch_size, 1, hidden_size * 2)
        attention_output = attention_output.squeeze(1)  # shape: (batch_size, hidden_size * 2)
        return attention_output, attention_weight


class IMDB_BIRNN_Attention(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 embedding_dim: int = 100,
                 hidden_dim: int = 256,
                 output_dim: int = 1,
                 n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=True,
                          dropout=0.5,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.attention = SelfAttention(2 * hidden_dim)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor):
        text = text.to(torch.int)  # shape: (batch_size, sentence_len)
        self.rnn.flatten_parameters()
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'), batch_first=self.rnn.batch_first)
        rnn_output, hidden = self.rnn(packed_embedded)
        hidden_output = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=self.rnn.batch_first)
        rescaled_hidden, attention_weight = self.attention(hidden_output, rnn_output, rnn_output)

        return self.fc(rescaled_hidden)

    def get_embedding(self, text: torch.Tensor):
        text = text.to(torch.int)
        return self.embedding(text)

    def forward_embedding(self, embedded: torch.Tensor, text_lengths: torch.Tensor):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'), batch_first=self.rnn.batch_first)
        rnn_output, hidden = self.rnn(packed_embedded)
        hidden_output = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=self.rnn.batch_first)
        rescaled_hidden, attention_weight = self.attention(hidden_output, rnn_output, rnn_output)
        return self.fc(rescaled_hidden)

    def get_attention_weight(self, text: torch.Tensor, text_lengths: torch.Tensor):
        text = text.to(torch.int)  # shape: (batch_size, sentence_len)
        self.rnn.flatten_parameters()
        embedded = self.embedding(text)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'), batch_first=self.rnn.batch_first)
        rnn_output, hidden = self.rnn(packed_embedded)
        hidden_output = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=self.rnn.batch_first)
        rescaled_hidden, attention_weight = self.attention(hidden_output, rnn_output, rnn_output)
        return attention_weight

    def eval_interpretability(self):
        self.eval()

        # cudnn RNN backward can only be called in training mode.
        self.rnn.train(True)
        self.rnn.dropout = 0.

        return self


class IMDB_BIRNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 embedding_dim: int = 100,
                 hidden_dim: int = 256,
                 output_dim: int = 1,
                 n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=True,
                          dropout=0.5,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor):
        text = text.to(torch.int)
        # text = [sent len, batch size]
        self.rnn.flatten_parameters()
        embedding_output = self.embedding(text)
        dropout_output = self.dropout(embedding_output)

        # embedded = [sent len, batch size, emb dim]
        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            dropout_output, text_lengths.to('cpu'), batch_first=self.rnn.batch_first)
        _, hidden = self.rnn(packed_embedded)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden_output = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden_output)

    def get_embedding(self, text):
        text = text.to(torch.int)
        return self.embedding(text)

    def forward_embedding(self, embedding_input, text_lengths):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedding_input, text_lengths.to('cpu'), batch_first=self.rnn.batch_first)
        _, hidden = self.rnn(packed_embedded)

        hidden_output = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden_output)

    def eval_interpretability(self):
        self.eval()

        # cudnn RNN backward can only be called in training mode.
        self.rnn.train(True)
        self.rnn.dropout = 0.

        return self


class IMDB_BILSTM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 embedding_dim: int = 100,
                 hidden_dim: int = 256,
                 output_dim: int = 1,
                 n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=True,
                           dropout=0.5,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor):
        text = text.to(torch.int)
        self.rnn.flatten_parameters()
        # text = [sent len, batch size]
        embedding_output = self.embedding(text)
        dropout_output = self.dropout(embedding_output)

        # embedded = [sent len, batch size, emb dim]
        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            dropout_output, text_lengths.to('cpu'), batch_first=self.rnn.batch_first)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden_output = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden_output)

    def get_embedding(self, text):
        text = text.to(torch.int)
        return self.embedding(text)

    def forward_embedding(self, embedding_input, text_lengths):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedding_input, text_lengths.to('cpu'), self.rnn.batch_first)
        _, (hidden, _) = self.rnn(packed_embedded)

        hidden_output = torch.cat(
            [hidden[-2], hidden[-1]],
            dim=1
        )
        return self.fc(hidden_output)

    def eval_interpretability(self):
        self.eval()

        # cudnn RNN backward can only be called in training mode.
        self.rnn.train(True)
        self.rnn.dropout = 0.
        self.dropout.eval()

        return self


class IMDB_GRU(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 embedding_dim: int = 100,
                 hidden_dim: int = 256,
                 output_dim: int = 1,
                 n_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=True,
                          dropout=0.5,
                          batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor):
        text = text.to(torch.int)
        # text = [sent len, batch size]
        self.rnn.flatten_parameters()
        embedding_output = self.embedding(text)
        dropout_output = self.dropout(embedding_output)

        # embedded = [sent len, batch size, emb dim]
        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            dropout_output, text_lengths.to('cpu'), batch_first=self.rnn.batch_first)
        _, hidden = self.rnn(packed_embedded)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden_output = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden_output)

    def get_embedding(self, text):
        text = text.to(torch.int)
        return self.embedding(text)

    def forward_embedding(self, embedding_input, text_lengths):
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedding_input, text_lengths.to('cpu'), batch_first=self.rnn.batch_first)
        _, hidden = self.rnn(packed_embedded)

        hidden_output = torch.cat(
            [hidden[-2], hidden[-1]],
            dim=1
        )
        return self.fc(hidden_output)

    def eval_interpretability(self):
        self.eval()

        # cudnn RNN backward can only be called in training mode.
        self.rnn.train(True)
        self.rnn.dropout = 0.

        return self


class IMDB_CNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 pad_idx: int,
                 embedding_dim: int,
                 n_filters: List[int],
                 filter_sizes: List[int]):
        super(IMDB_CNN, self).__init__()
        assert len(n_filters) == len(filter_sizes)
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        for _index_ in range(len(n_filters)):
            self.add_module(
                "conv_{}".format(_index_),
                nn.Conv2d(in_channels=1,
                          out_channels=n_filters[_index_],
                          kernel_size=(filter_sizes[_index_], embedding_dim))
            )
        self.register_buffer("n_submodule", torch.ones((1,), dtype=torch.int8) * len(n_filters))
        self.fc = nn.Linear(sum(n_filters), 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor):
        text = text.to(torch.int)
        # Noted: do not delete the `text_lengths` parameter.
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)  # shape: (batch_size, 1, seq_len, embedding_size)
        batch_size = embedded.shape[0]
        pool_results = []
        for _index_ in range(self.get_buffer("n_submodule").cpu().detach().item()):
            conv_module = self.get_submodule("conv_{}".format(_index_))  # type: nn.Conv2d
            filter_size = conv_module.kernel_size[0]
            conv_result = conv_module(embedded).squeeze(3)  # shape: (batch_size. channel_size, seq_len-filter_size+1)
            relu_result = F.relu(conv_result)
            sub_pool_results = []
            for batch_index in range(batch_size):
                sub_pool_results.append(
                    torch.mean(
                        relu_result[batch_index, :, :text_lengths[batch_index] - filter_size + 1], dim=1
                    )
                )
            sub_pool_results = torch.stack(sub_pool_results, dim=0)  # shape: (batch_size, channel_size)
            pool_results.append(sub_pool_results)
        cat = self.dropout(torch.cat(pool_results, dim=1))
        return self.fc(cat)

    def get_embedding(self, text):
        return self.embedding(text)

    def forward_embedding(self, embedding_input, text_lengths):
        embedded = embedding_input.unsqueeze(1)
        batch_size = embedded.shape[0]
        pool_results = []
        for _index_ in range(self.get_buffer("n_submodule").cpu().detach().item()):
            conv_module = self.get_submodule("conv_{}".format(_index_))  # type: nn.Conv2d
            filter_size = conv_module.kernel_size[0]
            conv_result = conv_module(embedded).squeeze(3)  # shape: (batch_size, channel_size, seq_len-filter+1)
            relu_result = F.relu(conv_result)
            sub_pool_results = []
            for batch_index in range(batch_size):
                sub_pool_results.append(
                    torch.mean(
                        relu_result[batch_index, :, :text_lengths[batch_index] - filter_size + 1], dim=1
                    )
                )
            sub_pool_results = torch.stack(sub_pool_results, dim=0)  # shape: (batch_size, channel_size)
            pool_results.append(sub_pool_results)
        cat = self.dropout(torch.cat(pool_results, dim=1))
        return self.fc(cat)


IMDB_MODEL_CLASS = Union[IMDB_BILSTM, IMDB_GRU, IMDB_CNN, IMDB_FCN]


class UDPOS_BERT(nn.Module):
    def __init__(self, bert, output_dim, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]
        text = text.permute(1, 0)
        # text = [batch size, sent len]
        embedded = self.dropout(self.bert(text)[0])
        # embedded = [batch size, seq len, emb dim]
        embedded = embedded.permute(1, 0, 2)
        # embedded = [sent len, batch size, emb dim]
        predictions = self.fc(self.dropout(embedded))
        # predictions = [sent len, batch size, output dim]
        return predictions

    def get_embedding(self, text):
        return self.bert(text)[0]

    def forward_embedding(self, embedding_input):
        embedded = embedding_input.permute(1, 0, 2)
        predictions = self.fc(embedded)
        return predictions


class UDPOS_BILSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,
                 n_layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            dropout=0.25 if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.25)

    def forward(self, text: torch.Tensor, text_index: int = None, batch_first: bool = False) -> torch.Tensor:
        """
        Noted: Take care of the parameters `text_index` and `batch_first`;
        For Training/Testing, use it like `model.forward(text)`;
        For Explaining the model, please use it like `model.forward(text, text_index=text_index, batch_first=True)`;
        :param text: model input.
        :param text_index: which word we want to check its' model prediction.
        :param batch_first: whether the model input is batch first.
        :return: model output
        """
        text_length = text.shape[1] if batch_first else text.shape[0]
        assert text_index is None or (isinstance(text_index, int) and 0 <= text_index <= text_length)
        # text = [sent len, batch size]
        # pass text through embedding layer
        embedded = self.dropout(self.embedding(text))
        if batch_first:  # the rnn layer has been set as `batch_first=False`
            embedded = embedded.permute(1, 0, 2)
        # pass embeddings into LSTM
        outputs, _ = self.lstm(embedded)
        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step
        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]
        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))  # shape: [sent len, batch size, output dim]
        if batch_first:
            predictions = predictions.permute(1, 0, 2)  # shape: [batch size, sent len, output dim]

        if text_index is not None:
            if batch_first:
                predictions = predictions[:, text_index]  # shape: [batch size, output dim]
            else:
                predictions = predictions[text_index]

        return torch.nn.functional.softmax(predictions, dim=-1)

    def get_embedding(self, text):
        return self.embedding(text)

    def forward_embedding(self, embedded, text_index: int = None, batch_first: bool = False):
        text_length = embedded.shape[1] if batch_first else embedded.shape[0]
        assert text_index is None or (isinstance(text_index, int) and 0 <= text_index <= text_length)
        if batch_first:  # the rnn layer has been set as `batch_first=False`
            embedded = embedded.permute(1, 0, 2)
        # pass embeddings into lstm
        outputs, _ = self.lstm(embedded)
        predictions = self.fc(outputs)
        if batch_first:
            predictions = predictions.permute(1, 0, 2)  # shape: [batch size, sent len, output dim]

        if text_index is not None:
            if batch_first:
                predictions = predictions[:, text_index]  # shape: [batch size, output dim]
            else:
                predictions = predictions[text_index]
        return torch.nn.functional.softmax(predictions, dim=-1)

    def eval_interpretability(self):
        self.eval()

        # cudnn RNN backward can only be called in training mode.
        self.lstm.train(True)
        self.lstm.dropout = 0.
        self.dropout.eval()

        return self


class UDPOS_BIRNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,
                 n_layers, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim,
                                      embedding_dim,
                                      padding_idx=pad_idx)
        self.rnn = nn.RNN(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=True,
                          dropout=0.25 if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.25)

    def forward(self, text: torch.Tensor, text_index: int = None, batch_first: bool = False):
        """
        Noted: Take care of the parameters `text_index` and `batch_first`;
        For Training/Testing, use it like `model.forward(text)`;
        For Explaining the model, please use it like `model.forward(text, text_index=text_index, batch_first=True)`;
        :param text: model input.
        :param text_index: which word we want to check its' model prediction.
        :param batch_first: whether the model input is batch-first.
        :return: model output
        """
        text_length = text.shape[1] if batch_first else text.shape[0]
        assert text_index is None or (isinstance(text_index, int) and 0 <= text_index <= text_length)
        # text = [sent len, batch size]
        # pass text through embedding layer
        embedded = self.dropout(self.embedding(text))
        if batch_first:  # the rnn layer has been set as `batch_first=False`.
            embedded = embedded.permute(1, 0, 2)
        # embedded = [sent len, batch size, emb dim]
        # pass embeddings into LSTM
        outputs = self.rnn(embedded)[0]
        # outputs holds the backward and forward hidden states in the final layer
        # hidden and cell are the backward and forward hidden and cell states at the final time-step
        # output = [sent len, batch size, hid dim * n directions]
        # hidden/cell = [n layers * n directions, batch size, hid dim]
        # we use our outputs to make a prediction of what the tag should be
        predictions = self.fc(self.dropout(outputs))
        if batch_first:
            predictions = predictions.permute(1, 0, 2)  # shape: [batch size, sent len, output dim]

        if text_index is not None:
            if batch_first:
                predictions = predictions[:, text_index]
            else:
                predictions = predictions[text_index]

        return torch.nn.functional.softmax(predictions, dim=-1)

    def get_embedding(self, text):
        return self.embedding(text)

    def forward_embedding(self, embedding_input, text_index: int = None, batch_first: bool = False):
        text_length = embedding_input.shape[1] if batch_first else embedding_input.shape[0]
        assert text_index is None or (isinstance(text_index, int) and 0 <= text_index <= text_length)
        if batch_first:  # the rnn layer has been set as `batch_first=False.`
            embedding_input = embedding_input.permute(1, 0, 2)
        outputs = self.rnn(embedding_input)[0]
        predictions = self.fc(outputs)
        if batch_first:
            predictions = predictions.permute(1, 0, 2)

        if text_index is not None:
            if batch_first:
                predictions = predictions[:, text_index]
            else:
                predictions = predictions[text_index]

        return torch.nn.functional.softmax(predictions, dim=-1)

    def eval_interpretability(self):
        self.eval()

        # cudnn RNN backward can only be called in training mode.
        self.rnn.train(True)
        self.rnn.dropout = 0.
        self.dropout.eval()

        return self


UDPOS_MODEL_CLASS = Union[UDPOS_BERT, UDPOS_BILSTM, UDPOS_BIRNN]
