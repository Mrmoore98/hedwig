import logging
import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.onnx

from common.evaluate import EvaluatorFactory
from common.train import TrainerFactory
from datasets.aapd import AAPDHierarchical as AAPD
from datasets.imdb import IMDBHierarchical as IMDB
from datasets.imdb_2 import IMDBHierarchical as IMDB_2
from datasets.imdb_stanford import IMDBHierarchical as IMDB_stanford
from datasets.reuters import ReutersHierarchical as Reuters
from datasets.yelp2014 import Yelp2014Hierarchical as Yelp2014
from models.oh_cnn_HAN.args import get_args
from models.oh_cnn_HAN.model import One_hot_CNN
from models.oh_cnn_HAN.one_hot_vector_preprocess import One_hot_vector

args = get_args()
config = deepcopy(args)
config.output_channel = 1000
config.input_channel = 30000
config.kernel_H = 1
config.kernel_W = 3
config.rnn_hidden_size = 100
config.target_class =2
config.max_size = 5
config.fill_value = 1


model = One_hot_CNN(config)
model = model.cuda()

input_simulate = torch.randint(0,config.max_size-1,(1,3,5)).cuda()

# scores_rounded = F.sigmoid(scores).round().long()
# predicted_labels.extend(scores_rounded.cpu().detach().numpy())
# target_labels.extend(torch.argmax(batch.label, dim=1).cpu().detach().numpy())
# total_loss += F.binary_cross_entropy_with_logits(scores, torch.argmax(batch.label, dim=1).float(), size_average=False).item()

result = model(input_simulate)
import pdb; pdb.set_trace()
#Test Success!!!!