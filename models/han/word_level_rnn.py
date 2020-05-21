import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WordLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        word_num_hidden = config.word_num_hidden
        words_num = config.words_num
        self.words_dim = config.words_dim
        self.mode = config.mode
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, self.words_dim).uniform(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            print("Unsupported order")
            exit()
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        # self.GRU = nn.GRU(self.words_dim, word_num_hidden, bidirectional=True)
        self.GRU = nn.LSTM(self.words_dim, word_num_hidden, bias=True, dropout=0.2, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()

        self.em_layer_norm = nn.LayerNorm(self.words_dim)
        self.gru_norm = nn.LayerNorm(word_num_hidden*2)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.CNN = config.cnn
        if self.CNN:
            self.cnn_layer = nn.ModuleList()
            self.kernel_set = [2,3,4]
            for kernel_size in self.kernel_set: 
                new_cnn = conv1d_same_padding( config.words_dim, 2 * word_num_hidden, kernel_size,\
                        stride=1, dilation=1, bias=True)
                self.cnn_layer.append(new_cnn)

            self.reduce_layer = nn.Linear(len(self.kernel_set)*word_num_hidden*2, word_num_hidden*2, bias=True)

    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else :
            print("Unsupported mode")
            exit()
        
        x = x*math.sqrt(self.words_dim)
        x = self.em_layer_norm(x)
        x = self.dropout(x)
        #x:[word numbers, batch size, word dim]

        if self.CNN:
            x = x.permute(1,2,0)
            cnn_output = []
            for cnn in self.cnn_layer:

                cnn_output.append(cnn(x))
            
            h = torch.cat(cnn_output, dim=1)
            #h :[bs, word dim, length]

            h = self.reduce_layer(h.permute(2,0,1))
        else:
            h, _ = self.GRU(x)

        h = self.gru_norm(h)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1)
        return x.transpose(1, 0).unsqueeze(0), None

def conv1d_same_padding(inputs, out_channels, kernel_size, bias=None, stride=1, dilation=1, groups=1):

  effective_filter_size = (kernel_size - 1) * dilation + 1
  outs = (inputs - effective_filter_size) // stride + 1

  padding = max(0, outs - inputs)
  odd = (padding % 2 != 0)

  if odd:
    inputs = F.pad(inputs, [0, 1])

  return nn.Conv1d(inputs, out_channels, bias, stride, padding= padding // 2, dilation=dilation, groups=groups)