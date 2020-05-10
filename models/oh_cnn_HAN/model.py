import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import time
import math

from models.oh_cnn_HAN.check_text import match_str

class One_hot_CNN(nn.Module):
    '''
    The input is one hot encodeing e.g.,0000010000.
    This Class is designed for learning n-gram like phrase-level representations.
    Notice:
        1.given computation resourse we need to compress our computational cost. For example, the CNN's weight martices need to be loaded dynamically.
        2.sparase operation may introduce in future.
        3.GRU is the simplified version of LSTM, so LSTM may achieve better performance when dataset is large.  

    '''

    def __init__(self, config):
        super(One_hot_CNN,self).__init__()  

        output_channel  = config.output_channel
    
        self.kernel_H   = config.kernel_H
        self.kernel_W   = config.kernel_W
        self.stride     = config.stride
        rnn_hidden_size = config.rnn_hidden_size
        
        self.fill_value = config.fill_value
        self.max_dim    = config.max_size
        self.zero_vec   = nn.Parameter(torch.zeros(1, self.max_dim), requires_grad = False)

        # self.kernel     = nn.ModuleList()

        #CNN Word Level
        self.kernel  = nn.ParameterList()
        self.bias    = nn.ParameterList()
        self.cnn_num = len(self.kernel_H) 
        for i in range(self.cnn_num):
            temp_kernel = torch.empty(output_channel[i], self.max_dim, self.kernel_H[i], self.kernel_W[i])
            nn.init.xavier_normal_(temp_kernel)
            self.kernel.append(nn.Parameter(temp_kernel, requires_grad = True))
            temp_bias = torch.empty(output_channel[i])
            temp_bias.uniform_(-0.25, 0.25)
            self.bias.append(nn.Parameter(temp_bias, requires_grad = True))   
        self.relu = nn.ReLU()
        
        # Sentence-Level
        output_dim = sum(output_channel)
        self.gru = torch.nn.GRU(input_size = output_dim, hidden_size= rnn_hidden_size, num_layers = 1, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc  = nn.Linear(rnn_hidden_size*2 ,  1 if config.is_binary else config.target_class, bias=True)
        # self.fc_2= nn.Linear(rnn_hidden_size*2 + output_channel*2 , 1 if config.is_binary else config.target_class)
        
        if config.attention:
            # Word Attention
            self.word_context_weights = nn.Parameter(torch.rand(output_dim, 1))
            self.word_att_linear      = nn.Linear(output_dim, output_dim, bias=True)
            self.word_context_weights.data.uniform_(-0.25, 0.25)
            self.soft_word            = nn.Softmax()
            # Sentence Attention
            self.sentence_context_weights = nn.Parameter(torch.rand(2 * rnn_hidden_size, 1))
            self.sentence_context_weights.data.uniform_(-0.1, 0.1)
            self.sentence_linear       = nn.Linear(2 * rnn_hidden_size, 2 * rnn_hidden_size, bias=True)
            self.soft_sentence         = nn.Softmax()

        # without RNN
        # self.fc1      = nn.Linear(output_channel*2, 1 if config.is_binary else config.target_class)#binary
        self.id       = config.id
        self.use_RNN  = config.use_RNN
        self.tot_time = 0
        self.scale_factor_W = math.sqrt(output_dim)
        self.scale_factor_S = math.sqrt(2 * rnn_hidden_size)


    def forward(self, input):
        '''
                   010 110 110
                        |
                    conv layer
                        |
                    Lstm layer
                        |
            reparameterization(optional)
              
            INPUT: Batch_size x (V) extra process x  Sentence_Num  x Sentence_Max_Len
        '''        
        # truncate kernel for saving resource
        batch_size    = input.shape[0]
        sentence_num  = input.shape[1]
        if len(input.shape)>2:
            sentence_size = input.shape[2]
        # else: 
        #     sentence_num = 10 if sentence_num>10 
        #     input = input.contiguous().view(batch_size, sentence_num, -1)
          
        time_1 = time.time()    
        input, weight, compress_size = getattr(self,'ver{}'.format(self.id))(input)
        time_1 = time.time() - time_1

        if self.use_RNN:
            time_2 = time.time()
            self.cnn_out = []
            for i in range(self.cnn_num):
                out  = self.same_pad_conv2d(input, weight[i], self.bias[i], self.stride[i])
                out  = self.relu(out)
                # out = F.adaptive_max_pool2d(out, output_size = (sentence_num, 1)).squeeze(-1)
                # out = out.permute(0,2,1) # output: batch, sentence_num, input size
                self.cnn_out.append(out)
        
            concat_out = torch.cat(self.cnn_out, dim = 1).permute(0,2,3,1)
            word_att   = self.word_attention(concat_out)
            # concat_out   = F.adaptive_avg_pool1d(concat_out.permute(0,2,1), output_size = 1).squeeze(-1)
            time_2 = time.time() - time_2
            time_3 = time.time()

            gru_out, h_n = self.gru(word_att)
            sen_att      = self.sen_attention(gru_out)
            # rnn_pool_out = F.adaptive_avg_pool1d(gru_out.permute(0,2,1), output_size = 1).squeeze(-1)
            # output       = self.relu(rnn_pool_out)

            # concat_out   = F.adaptive_avg_pool1d(concat_out.permute(0,2,1), output_size = 1).squeeze(-1)
            # output       = torch.cat((concat_out, output), dim=-1)
            # output       = self.fc(output)
            output       = self.fc(sen_att).squeeze(-1)

            time_3 = time.time() - time_3
            self.tot_time += time_1+time_2+time_3
            # print("1:{}, 2:{}, 3:{}, tot:{}".format(time_1/tot_time,time_2/tot_time,time_3/tot_time,tot_time))
        else:
            input     = input.view(batch_size, compress_size, 1,-1)
            self.cnn_out = []
            for i in range(self.cnn_num):
                out    = self.F.conv2d(input, weight[i], self.bias[i]).squeeze(-2)
                out    = self.relu(out)
                out = F.adaptive_max_pool1d(out, output_size = 1).squeeze(-1)
                self.cnn_out.append(out)
                    
            combi_out = torch.cat(self.cnn_out,dim=1)
            output    = self.fc1(combi_out).squeeze(-1)

        return output

    def same_pad_conv2d(self, input, kernel, bias, stride=1,  dilation=1, groups=1):
        '''preserved height&width'''
        input_size_H = input.shape[2]
        input_size_W = input.shape[3]
        kernel_H     = kernel.shape[2]
        kernel_W     = kernel.shape[3]    
        # effective_filter_size_rows = (kernel_H - 1) * dilation[0] + 1
        # effective_filter_size_rows = (kernel_W - 1) * dilation[0] + 1

        tmp_output_size   = (input_size_H + stride - 1) // stride
        padding_needed_H  = max(0, (tmp_output_size - 1) * stride + kernel_H - input_size_H)
        h_odd = (padding_needed_H %2 !=0)

        tmp_output_size   = (input_size_W + stride - 1) // stride
        padding_needed_W  = max(0, (tmp_output_size - 1) * stride + kernel_W - input_size_W)
        w_odd = (padding_needed_W % 2 != 0)

        if h_odd or w_odd:
           input = F.pad(input, [0, int(w_odd), 0, int(h_odd)])#reverse 

        return F.conv2d(input, kernel, bias, stride, padding = (int(padding_needed_H)//2, int(padding_needed_W)//2), dilation=dilation, groups=groups)

    def ver1(self, input):
        '''Adaptively adjust weight to accommodate input'''        
        # allocate weight 
      
        index_weight  = torch.unique(input).type(torch.cuda.LongTensor)
        compress_size = index_weight.shape[0] 
        
        weight = []
        for i in range(self.cnn_num):
            weight.append(self.kernel[i][:,index_weight,:,:])
            

        index_one_hot = input.type(torch.cuda.LongTensor).unsqueeze(1)

        output        = self.zero_vec.repeat(*input.shape, 1).permute(0,3,1,2)
        # import pdb; pdb.set_trace()
        # output        = torch.zeros(*input.shape, self.max_dim).permute(0,3,1,2).cuda()

        output.scatter_(1, index_one_hot, self.fill_value)
        compressed_output = output[:, index_weight, :, :]
       

        return compressed_output, weight, compress_size
    
    def ver2(self, input):
        '''leave the weight alone, focus on generating one hot encoding '''
        weight = []
        for i in range(self.cnn_num):
            weight.append(self.kernel[i])
        
        output    = self.zero_vec.repeat(*input.shape, 1).permute(0,3,1,2)
    
        # output = torch.zeros(*input.shape, self.max_dim).permute(0,3,1,2).cuda()
        # onehot coding generating    
        output.scatter_(1, input.type(torch.LongTensor).unsqueeze(1).cuda(),self.fill_value)
        return output, weight, self.max_dim

    def ver3(self, input):
        '''Adaptively adjust weight to accommodate input
            for one sequence-style input
        '''        
        # allocate weight 
      
        index_weight  = torch.unique(input).type(torch.cuda.LongTensor)
        compress_size = index_weight.shape[0] 

        weight = []
        for i in range(self.cnn_num):
            weight.append(self.kernel[i][:,index_weight,:,:])
        
        index_one_hot = input.type(torch.cuda.LongTensor).unsqueeze(1)

        output        = self.zero_vec.repeat(*input.shape, 1).permute(0,2,1)
        # import pdb; pdb.set_trace()
        # output        = torch.zeros(*input.shape, self.max_dim).permute(0,3,1,2).cuda()

        output.scatter_(1, index_one_hot, self.fill_value)
        compressed_output = output[:, index_weight, :]
       

        return compressed_output, weight, compress_size

    def word_attention(self, h):
        # h : bs, sennum, senlen, word dim

        x = torch.tanh(self.word_att_linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=-1)
        x = F.softmax( x/self.scale_factor_W, dim=-1 )

        x = torch.mul(h.permute(3,0,1,2), x).permute(1,2,3,0)
        x = torch.sum(x, dim=2)

        return x 

    def sen_attention(self, input_tensor):
        # input tensor: bs sennum word_dim

        x = torch.tanh(self.sentence_linear(input_tensor))
        x = torch.matmul(x, self.sentence_context_weights)
        # x: bs, sennum, 1
        x = x.squeeze(dim=-1)
        x = F.softmax( x/self.scale_factor_S, dim= -1)
        x = torch.mul( input_tensor.permute(2, 0, 1), x).permute(1, 2, 0)
        x = torch.sum( x, dim=1)

        return x
    




        

