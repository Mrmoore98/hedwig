import datetime
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle

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
        self.real_min = 2.2e-16
        self.eulergamma = 0.5772
        self.eps = None
        
        with open(data_path, 'rb') as file:
            deconv_kernel = pickle.load(file)

        deconv_kernel = torch.from_numpy(deconv_kernel).reshape(1000, 30000, 3, 1)
        self.deconv_kernel = nn.parameter(deconv_kernel, requires_grad = False)
        self.zero_vec = nn.Parameter(torch.zeros(1, config.vae_word_dim), requires_grad = False)
        self.shape_scale_cnn = nn.Conv2d(config.word_hidden_dim*2, 2*2*config.word_hidden_dim, 1, groups=2)
        self.word_hidden_dim = config.word_hidden_dim
    
    def ELBO(self, scores, label, W, origin_data):
        
        W = self.shape_scale_cnn(W)
        Wei_shape, Wei_scale = W[:,:self.word_hidden_dim*2,:,:], W[:,self.word_hidden_dim*2:,:,:]
        Gam_shape = torch.ones_like(Wei_shape)
        Gam_scale = torch.ones_like(Wei_scale)
        CE_loss = self.cross_entropy(scores, label)
        KL_loss = self.KL_GamWei(Gam_shape, Gam_scale, Wei_shape, Wei_scale)
        theta = self.reparameterization(Wei_shape, Wei_scale)
        Likelihood = self.Likeihood(self.Params, theta, origin_data)
        Loss = CE_loss - Likelihood + KL_loss
        return Loss

    def log_max(self, input):
        return torch.log(torch.max(input, self.real_min))

    def reparameterization(self, Wei_shape, Wei_scale):
        self.eps = torch.empty_like(Wei_shape, dtype= torch.float64 ).uniform_(from=0, to=1)
        theta = Wei_scale * tf.pow(-self.log_max(1 - eps), 1 / Wei_shape)
        return theta

    def KL_GamWei(self, Gam_shape, Gam_scale, Wei_shape, Wei_scale):  # K_dim[i] * none
        KL_Part1 = self.eulergamma * (1 - 1 / Wei_shape) + self.log_max(Wei_scale / Wei_shape) + 1 + Gam_shape * self.log_max(Gam_scale)
        KL_Part2 = -1*torch.lgamma(Gam_shape) + (Gam_shape - 1) * (self.log_max(Wei_scale) - self.eulergamma / Wei_shape)
        KL = KL_Part1 + KL_Part2 - Gam_scale * Wei_scale * tf.exp(torch.lgamma(1 + 1 / Wei_shape))
        return KL

    def Likeihood(self, Params, theta, origin_data):
        likelihood = 0
        Orgin_X, weight = self.to_oh(origin_data)
        PhiTheta_1 = self.deconv2d(theta, weight, Orgin_X.size(2), Orgin_X.size(3))
        E_q = Orgin_X * log_max(PhiTheta_1) - PhiTheta_1 - torch.lgamma(Orgin_X + 1)
        likelihood = torch.sum(E_q)
        
        return likelihood

    def deconv2d(self, input, weight, expected_W, expected_H, stride=(1,1), dilation=(1,1), padding=(0,0)):
        
        output_pad = (0,0)
        output_pad[0] = expected_W - (input.size(2)-1)*stride[0] + 2*padding[0]- dilation[0]*(self.weight.size(0)-1) - 1
        output_pad[1] = expected_H - (input.size(3)-1)*stride[1] + 2*padding[1]- dilation[1]*(self.weight.size(1)-1) - 1
        output = F.conv_transpose2d(input, weight, output_padding = output_pad, bias = True)
        return output

    def to_oh(self, input):
        '''Adaptively adjust weight to accommodate input'''        
        # allocate weight 
        index_weight  = torch.unique(input).type(torch.cuda.LongTensor)
        weight=self.self.deconv_kernel[:,index_weight,:,:])
        index_one_hot = input.type(torch.cuda.LongTensor).unsqueeze(1)
        output        = self.zero_vec.repeat(*input.shape, 1).permute(0,3,1,2)
        output.scatter_(1, index_one_hot, self.fill_value)
        compressed_output = output[:, index_weight, :, :]

        return compressed_output, weight



if __name__ == "__main__":
    
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = Loss.apply('mse')(input, target)
    output.backward()