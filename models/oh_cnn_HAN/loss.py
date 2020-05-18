import datetime
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

class Loss(object):

    def __init__(self):
        pass

    @classmethod
    def apply(cls, name):

        return getattr(cls,name)

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



if __name__ == "__main__":
    
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = Loss.apply('mse')(input, target)
    output.backward()