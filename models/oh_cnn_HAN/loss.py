import datetime
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
from models.oh_cnn_HAN.label_smooth import LabelSmoothing
import sys

class Loss(nn.Module):

    def __init__(self, config):
        super(Loss, self).__init__()
        
        if config.vae_struct:
           self.vaeloss = VAELoss(config)

    def apply(self, name):
        return getattr(self,name)

    def cross_entropy(self, scores, label):

        if len(scores.shape)<2:
            scores = scores.unsqueeze(0)
        n_correct = 0
        predictions  = torch.argmax(scores, dim=-1).long().cpu().numpy()
        ground_truth = torch.argmax(label, dim=-1).cpu().numpy()
        n_correct   += np.sum(predictions == ground_truth)
        loss = F.cross_entropy(scores, torch.argmax(label, dim=-1))

        return loss
    
    def binary_cross_entropy(self, scores, label):

        predictions  = F.sigmoid(scores).round().long().cpu().numpy()
        ground_truth = torch.argmax(label, dim=1).cpu().numpy()
        n_correct   += np.sum(predictions == ground_truth)

        loss = F.binary_cross_entropy_with_logits(scores, torch.argmax(label, dim=1).type(torch.cuda.FloatTensor))
        return loss

    def mse(self, scores, label):

        target_len   = label.shape[1]
        predictions  = F.sigmoid(scores)*target_len
        predictions  = predictions.int().cpu().numpy()
        ground_truth = torch.argmax(label, dim=1).cpu().numpy()
        n_correct   += np.sum(predictions == ground_truth)

        loss = F.mse_loss(scores, torch.argmax(label, dim=1).type(torch.cuda.FloatTensor))
        return loss
    
    def ELBO(self, scores, label, W, origin_data):

        return self.vaeloss.ELBO(scores, label, W, origin_data)


class VAELoss(nn.Module):

    def __init__(self, config):
        super(VAELoss, self).__init__()  
        if False:
            self.real_min = 2.2e-16
            self.eulergamma = 0.5772
            self.eps = None
            data_path = '/home/s/CNN-BiLSTM2/hedwig-data/Conv_PGDS_Batch_5_11.pkl'
            with open(data_path, 'rb') as file:
                deconv_kernel = pickle.load(file)
            deconv_kernel['D'] = np.concatenate((deconv_kernel['D'],np.random.rand(1000,1,3)),axis=1)   
            deconv_kernel = torch.from_numpy(deconv_kernel['D']).reshape(1000, 30000, 3, 1).permute(1,0,2,3)
            self.deconv_kernel = nn.Parameter(deconv_kernel, requires_grad = False)
            self.zero_vec = nn.Parameter(torch.zeros(1, config.vae_word_dim), requires_grad = False)
            self.shape_scale_cnn = nn.Conv2d(config.word_num_hidden*2, 2*1000, 1, groups=2)
            self.word_num_hidden = config.word_num_hidden
            self.bias = nn.Parameter(torch.empty(config.vae_word_dim, dtype= torch.float64).uniform_(-0.1, 0.1))
            self.fill_value = 1
            self.label_smoothing = LabelSmoothing(config)
            self.label_smooth = config.label_smoothing
    
    def ELBO(self, scores, label, W, origin_data):
        
        W = self.shape_scale_cnn(W)
        Wei_shape, Wei_scale = W[:,:W.shape[1]//2,:,:], W[:,W.shape[1]//2:,:,:]
        Gam_shape = torch.ones_like(Wei_shape)
        Gam_scale = torch.ones_like(Wei_scale)
        if self.label_smooth:
            CE_loss = self.label_smoothing(scores, label)
        else:
            CE_loss = self.cross_entropy(scores, label)
        KL_loss = self.KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale)
        theta = self.reparameterization(Wei_shape, Wei_scale)
        Likelihood = self.Likeihood(theta, origin_data)
        Loss = CE_loss - Likelihood + KL_loss
        return Loss

    def log_max(self, input):
        return torch.log(torch.max(input, torch.tensor([self.real_min], dtype=input.dtype)))

    def reparameterization(self, Wei_shape, Wei_scale):
        self.eps = torch.empty_like(Wei_shape, dtype= torch.float64 ).uniform_(0,1)
        theta = Wei_scale * torch.pow(-self.log_max(1 - self.eps), 1 / Wei_shape)
        return theta

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape, Wei_scale):  # K_dim[i] * none
        KL_Part1 = self.eulergamma * (1 - 1 / Wei_shape) + self.log_max(Wei_scale / Wei_shape) + 1 + Gam_shape * self.log_max(Gam_scale)
        KL_Part2 = -1*torch.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max(Wei_scale) - self.eulergamma / Wei_shape)
        KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * torch.exp(torch.lgamma(1 + 1 / Wei_shape))
        return torch.sum(KL)

    def Likeihood(self, theta, origin_data, Param=None):
        likelihood = 0
        Orgin_X, weight, bias = self.to_oh(origin_data)
        PhiTheta_1 = self.same_pad_conv2d(theta, weight, bias)
        E_q = Orgin_X * self.log_max(PhiTheta_1) - PhiTheta_1 - torch.lgamma(Orgin_X + 1)
        likelihood = torch.sum(E_q)
        
        return likelihood

    def deconv2d(self, input, weight, expected_W, expected_H, stride=(1,1), dilation=(1,1), padding=(0,0)):
        
        output_pad = [None, None]
        output_pad[0] = (input.size(2)-1)*stride[0] - expected_W - 2*padding[0] + dilation[0]*(weight.size(0)-1) + 1
        output_pad[1] = ((input.size(3)-1)*stride[1] - expected_H + dilation[1]*(weight.size(1)-1) + 1)/2
        
        output = F.conv_transpose2d(input, weight, output_padding = output_pad)
        return output

    def to_oh(self, input):
        '''Adaptively adjust weight to accommodate input'''        
        # allocate weight 
        index_weight  = torch.unique(input)
        weight=self.deconv_kernel[index_weight,:,:,:]
        bias  =self.bias[index_weight] 
        index_one_hot = input.unsqueeze(1)
        output        = self.zero_vec.repeat(*input.shape, 1).permute(0,3,1,2)
        output.scatter_(1, index_one_hot, self.fill_value)
        compressed_output = output[:, index_weight, :, :]

        return compressed_output, weight, bias
    
    def cross_entropy(self, scores, label):
        loss = F.cross_entropy(scores, torch.argmax(label, dim=-1))

        return loss
    
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



if __name__ == "__main__":
    import os
    sys.path.append(os.getcwd())
    from models.oh_cnn_HAN.args import get_args
    from copy import deepcopy
    args = get_args()
    config = deepcopy(args)
    config.word_num_hidden = 100
    config.sentence_num_hidden = 100
    config.target_class = 10
    config.vae_struct = True
    config.residual = False
    config.dropout_rate =0.5
    config.vae_word_dim = 30000
    
    aa = Loss(config)
    sentence_len = 20
    sentence_num = 10
    batch_size = 2
    input = torch.randn(batch_size, sentence_num, requires_grad=True)
    target = torch.randn(batch_size, sentence_num)
    origin_data_index = torch.randint(0,30000,(batch_size, sentence_num, sentence_len))
    word_vec = torch.randn(batch_size, sentence_num, sentence_len, 2*config.word_num_hidden, requires_grad=True).permute(0,3,1,2)
    output = aa.apply('ELBO')(input, target, word_vec, origin_data_index)
    output.backward()
    import pdb; pdb.set_trace()