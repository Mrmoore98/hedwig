import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, config):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss()
        self.confidence = 1.0 - config.smoothing
        self.smoothing = config.smoothing
        # self.size = size
        self.true_dist = None
        self.mode = config.ls_mode
        self.std  = config.std
        self.target_class = config.target_class
        self.x = torch.linspace(0., self.target_class, steps= self.target_class).cuda()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, target):
        return getattr(self, self.mode)(self.log_softmax(x), target)

    def origin(self,x,target):

        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (x.size(1) - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        self.true_dist = true_dist
        return self.criterion(x, Variable(self.true_dist, requires_grad=False))

    def normal_dis(self, x, target):

        self.true_dist = self.normal(target.unsqueeze(1).float())
        return self.criterion(x, Variable(self.true_dist, requires_grad=False)) 

    def normal(self, mean):

        x = self.x.repeat(mean.size(0),1)
        std = mean.data.clone().fill_(self.std)
        x = (x-mean)/std
        x = -1*torch.pow(x,2)/2 
        # import pdb; pdb.set_trace()
        z = torch.sum( torch.exp(x), dim=1, keepdim=True)
        dist = torch.exp(x)/ z
        # import pdb; pdb.set_trace()

        return dist 


if __name__ == "__main__":
    
    crit= LabelSmoothing(config)
    # mean = torch.tensor([ 1., 0.5]).reshape(2,1)
    # std  = torch.tensor([0.5, 2. ]).reshape(2,1)
    # plt.plot(np.linspace(0,10,num=10), crit.normal(mean,(2, 10))[0].numpy())
    # plt.show()


    def loss(x):
        d = x + 4
        predict = torch.FloatTensor([[1/d, x / d, 1 / d, 1 / d, 1 / d],])
        return crit(Variable(predict.log()), Variable(torch.LongTensor([1]))).item()
                
    
    # loss(1)
    # import pdb; pdb.set_trace()
    # aa = [loss(x) for x in range(1, 100)]
    # import pdb; pdb.set_trace()
    plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
    plt.show()

