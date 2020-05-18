import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class NoamOpt(object):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))


if __name__ == "__main__":
    

    ow_factor = 6
    ow_warmup = 30000
    ow_model_size = 300
  
        
    tot_step = 50000
    opts = [NoamOpt( ow_model_size, ow_factor, ow_warmup, None), 
            NoamOpt(300, 3, 8000, None),
            NoamOpt(600, 3, 9000, None)]

    plt.plot(np.arange(1, tot_step), [[opt.rate(i) for opt in opts] for i in range(1, tot_step)])
    plt.legend(["A", "B", "C"])
    plt.show()
        
