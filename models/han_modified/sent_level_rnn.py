import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
class SentLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.residual=config.residual
        if config.residual:
            sentence_num_hidden = config.word_num_hidden
        else:
            sentence_num_hidden = config.sentence_num_hidden
        word_num_hidden = config.word_num_hidden
        target_class = config.target_class


        self.sentence_context_weights = nn.Parameter(torch.rand(2 * sentence_num_hidden, 1))
        self.sentence_context_weights.data.uniform_(-0.1, 0.1)
        self.sentence_gru = nn.GRU(2 * word_num_hidden, sentence_num_hidden, bidirectional=True)
        self.sentence_linear = nn.Linear(2 * sentence_num_hidden, 2 * sentence_num_hidden, bias=True)
        self.fc = nn.Linear(2 * sentence_num_hidden , target_class)
        self.fc_cat = nn.Linear(2*2 * sentence_num_hidden , target_class)

        self.soft_sent = nn.Softmax()
        self.scale_factor = math.sqrt(2 * sentence_num_hidden)
        
 

        self.sen_vec_norm = nn.LayerNorm(config.word_num_hidden*2)
        if config.residual:
            self.SenGruRes = Residual_Block(config, sentence_num_hidden*2)
            self.SenAttRes = Residual_Block(config, sentence_num_hidden*2)
        # self.SenFfRes = Residual_Block(config, sentence_num_hidden*2)
        self.SenFfCat = Concate_Block(config, sentence_num_hidden*2)

        self.ff = PositionwiseFeedForward(sentence_num_hidden*2, sentence_num_hidden*2*4, config.dropout_rate)
        
    def forward(self, sen_vec):

        #sen_vec[seq_len, bs, word_num_hidden*2]
        # import pdb; pdb.set_trace()
        sen_vec = self.sen_vec_norm(sen_vec)
    
        if self.residual:
            # sentence_h = self.SenGruRes(sen_vec, self.sentence_gru)
            sentence_h,_ = self.sentence_gru(sen_vec)
            x = self.SenAttRes(sentence_h, self.han_attention)
            x = self.SenFfCat(x,self.ff)

            # x = self.ff(x)
            x = torch.sum(x, dim=0)
            x = self.fc_cat(x)

        else:
            sentence_h,_ = self.sentence_gru(sen_vec)
            x = self.han_attention(sentence_h)
            x = self.SenFfCat(x,self.ff)
        
            x = torch.sum(x, dim=0)
            x = self.fc_cat(x)
            
        # x = torch.sum(x, dim=0)
        # x = self.fc(x)
        
        return x
    
    def han_attention(self, input_tensor):

        x = torch.tanh(self.sentence_linear(input_tensor))
        x = torch.matmul(x, self.sentence_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_sent(x.transpose(1,0)/self.scale_factor)
        
        x = torch.mul(input_tensor.permute(2, 0, 1), x.transpose(1, 0)).permute(1, 2, 0)

        return x

    def self_attention(self, query, key, value):

        #MatMul&Scale
        x = torch.matmul(query, key.permute(-1,-2))/math.sqrt(query.size(-1))
        #Softmax
        x = F.softmax(x, dim=-1)
        #Matmul with value
        x = self.dropout(x)
        x = torch.matmul(x,value)
        return x

        



class Residual_Block(nn.Module):
    '''implement residual connection'''
    def __init__(self, config, input_size):
        super(Residual_Block, self).__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_normalize = nn.LayerNorm(input_size)

    def forward(self, input_data, sublayer):
        sublayer_input = self.layer_normalize(input_data)
        sublayer_output = sublayer(input_data)
        if isinstance(sublayer_output,tuple):
            # GRU output include hidden layer weight
            sublayer_output = sublayer_output[0]
        return input_data + self.dropout(sublayer_output)

class Concate_Block(nn.Module):
    '''implement concate connection'''
    def __init__(self, config, input_size):
        super(Concate_Block, self).__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_normalize = nn.LayerNorm(input_size)

    def forward(self, input_data, sublayer):
        sublayer_input = self.layer_normalize(input_data)
        sublayer_output = sublayer(input_data)
        if isinstance(sublayer_output,tuple):
            # GRU output include hidden layer weight
            sublayer_output = sublayer_output[0]
        return torch.cat((input_data,self.dropout(sublayer_output)),dim=2)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation. Point-wise feed forward"
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.w_1(x))
        x = self.w_2(self.dropout(x))
        return x