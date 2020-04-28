import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import time



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
        kernel_W        = config.kernel_W
        rnn_hidden_size = config.rnn_hidden_size
        
        self.fill_value = config.fill_value
        self.max_dim    = config.max_size
        self.zero_vec   = nn.Parameter(torch.zeros(1, self.max_dim), requires_grad = False)
        self.kernel_1   = torch.empty(output_channel, self.max_dim, self.kernel_H, kernel_W)
        nn.init.xavier_normal_(self.kernel_1)
        self.kernel_1   = nn.Parameter(self.kernel_1, requires_grad = True)
        self.relu       = nn.ReLU()

        # sentence-level
        self.gru = torch.nn.GRU(input_size=output_channel, hidden_size= rnn_hidden_size, num_layers = 1, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc  = nn.Linear(rnn_hidden_size*2, config.target_class)
        
        # without RNN
        self.fc1 = nn.Linear(output_channel, config.target_class-1)#binary

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
        sentence_size = input.shape[2]
        stride        = 1
        id            = 1

        use_RNN       = False

        input, weight_1, compress_size = getattr(self,'ver{}'.format(id))(input)

        if use_RNN:
            conv1_out    = F.conv2d(input, weight_1, padding = self.height_same(input, self.kernel_H, sentence_num, stride))
            conv1_out    = self.relu(conv1_out)
            pool1_out    = F.adaptive_max_pool2d(conv1_out, output_size = (sentence_num, 1)).squeeze(-1)
            pool1_out    = pool1_out.permute(0,2,1) # output: batch, sentence_num, input size
            gru_out, h_n = self.gru(pool1_out)
            pool2_out    = F.adaptive_avg_pool1d(gru_out.permute(0,2,1), output_size = 1).squeeze(-1)
            output       = self.fc(pool2_out)
        else:
            input     = input.view(batch_size, compress_size, 1,-1)
            conv1_out = F.conv2d(input, weight_1).squeeze(-2)
            conv1_out = self.relu(conv1_out)
            pool1_out = F.adaptive_max_pool1d(conv1_out, output_size = 1).squeeze(-1)
            output    = self.fc1(pool1_out).squeeze()

        return output

    def height_same(self, input, kernel_H, output_channel, stride=1):
        '''Only preserved height'''
        input_size = input.shape[1]
        tmp_output_size = (input_size + stride - 1) / stride
        padding_needed  = max(0, (tmp_output_size - 1) * stride + kernel_H - input_size)
    
        return (padding_needed, 0)

    def ver1(self, input):
        '''Adaptively adjust weight to accommodate input'''        
        # allocate weight 
      
        index_weight  = torch.unique(input).type(torch.cuda.LongTensor)
        compress_size = index_weight.shape[0] 
        weight_1      = self.kernel_1[:,index_weight,:,:]
        index_one_hot = input.type(torch.cuda.LongTensor).unsqueeze(1)

        output        = self.zero_vec.repeat(*input.shape, 1).permute(0,3,1,2)
        # import pdb; pdb.set_trace()
        # output        = torch.zeros(*input.shape, self.max_dim).permute(0,3,1,2).cuda()

        output.scatter_(1, index_one_hot, self.fill_value)
        compressed_output = output[:, index_weight, :, :]
       

        return compressed_output, weight_1, compress_size
    
    def ver2(self, input):
        '''leave the weight alone, focus on generating one hot encoding '''
        weight_1  = self.kernel_1
        
        output    = self.zero_vec.repeat(*input.shape, 1).permute(0,3,1,2)
    
        # output = torch.zeros(*input.shape, self.max_dim).permute(0,3,1,2).cuda()
        # onehot coding generating    
        output.scatter_(1, input.type(torch.LongTensor).unsqueeze(1).cuda(),self.fill_value)
        return output, weight_1, self.max_dim




        

