import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from models.oh_cnn_HAN.TCN import TemporalConvNet
# from models.oh_cnn_HAN.PCN import PyramidConvNet

class WordLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_num_hidden = config.word_num_hidden
        self.words_dim = config.words_dim
        self.vae_struct = config.vae_struct
        self.frontend_cnn = config.frontend_cnn
        self.CNN = config.cnn
        # han attention
        self.word_context_weights = nn.Parameter(torch.rand(2 * self.word_num_hidden, 1))
        self.linear = nn.Linear(2 * self.word_num_hidden, 2 * self.word_num_hidden, bias=True)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax(dim=-1)
        # RNN
        # self.GRU = nn.GRU(self.words_dim, self.word_num_hidden, bidirectional=True)
        self.GRU = nn.LSTM(self.words_dim, self.word_num_hidden, bidirectional=True, batch_first=True)
        if self.CNN:
            cnn_out_channels=np.linspace(config.words_dim, self.word_num_hidden, num=int(np.log2(config.min_seq_len)))
            self.TCN = TemporalConvNet(config.words_dim, list(cnn_out_channels)[1:], 2, 0.2)
            # PyramidConvNet()
        if self.frontend_cnn:
            self.front_cnn = conv1d_same_padding( config.words_dim, config.words_dim, 3, stride=1, dilation=1, bias=True)

        # Regularizatiom
        self.em_layer_norm = nn.LayerNorm(self.words_dim)
        # self.word_norm = nn.BatchNorm1d(self.word_num_hidden*2)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.em_dropout = nn.Dropout(config.dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x*math.sqrt(self.words_dim)
        x = self.em_dropout(x)
        #x:[word numbers, batch size, word dim]
        if self.CNN:
            x = x.permute(0,2,1)
            h = self.TCN(x)
            #h :[bs, word dim, length]   
            h = h.permute(0,1,2)
        else:
            if self.frontend_cnn:
                x = x.permute(0,2,1)
                x = self.front_cnn(x)
                x = self.relu(x)
                x = x.permute(0,1,2)
            x = self.em_layer_norm(x)
            h, _ = self.GRU(x)
        word_vec = h.unsqueeze(0) if self.vae_struct else None
        #h: [bs, length, word dim]
        x = self.han_attention(h)
        return x, word_vec

    def han_attention(self, h):
        x = h
        x = torch.tanh(self.linear(x))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        # x: [bs, length]
        x = self.soft_word(x)
        x = torch.mul(h.permute(2, 0, 1), x)
        x = torch.sum(x, dim=-1)
        # x: [word dim, bs]
        x = x.transpose(1, 0).unsqueeze(0)
        return x

def conv1d_same_padding(inputs, out_channels, kernel_size, bias=None, stride=1, dilation=1, groups=1):

    effective_filter_size = (kernel_size - 1) * dilation + 1
    outs = (inputs - effective_filter_size) // stride + 1

    padding = max(0, outs - inputs)
    odd = (padding % 2 != 0)

    if odd:
        inputs = F.pad(inputs, [0, 1])

    return nn.Conv1d(inputs, out_channels, kernel_size, stride, padding= padding // 2, dilation=dilation, groups=groups,bias=bias)