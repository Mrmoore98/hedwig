import csv
import os 
# import spacy
import re
import time
import numpy as np
import random
from collections import Counter
# spacy=spacy.load('en')
import pickle
from torchtext.vocab import Vectors
import logging
import torch

logger = logging.getLogger(__name__)

class One_hot_vector(Vectors):

    def __init__(self):

        self.itos = []
        self.stoi = {}
        self.vectors = None
        self.dim = 30000
        self.init_vec = torch.zeros(self.dim)
        self.fill_value = 5
        temp = torch.zeros(self.dim)
        temp[-1] = self.fill_value
        self.unk_init_vector = temp
        self.vector_init()
       
    def clean_string(self, string):
        """
        Performs tokenization and string cleaning for the Reuters dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.lower().strip().split()

    def unk_init(self, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.zeros(tensor.size())[-1] = 1
            # cls.cache[size_tup]
        return cls.cache[size_tup]

    def __getitem__(self, token):
        if token in self.stoi:
            return self.init_vec.scatter_(0,torch.LongTensor([self.stoi[token]]), self.fill_value)
        else:
            return self.unk_init_vector


    def vector_init(self):

        data_path = '/home/zhangxin/sentiment_analysis/hedwig-data/datasets/aclImdb/' 
        output_data_path = '/home/zhangxin/sentiment_analysis/hedwig-data/datasets/IMDB_stanford/' 

        ''' Acutally, we don't need to conduct any data in here. 
            Since this Class are designed for storing the word embedding which doesn't exist in our model. 
        '''
        # path_pt =output_data_path+"Vocabulary_matrix.pt"
        # if os.path.isfile(path_pt):
        #     logger.info('Loading vectors from {}'.format(path_pt))
        #     self.itos, self.stoi, self.vectors, self.dim = torch.load(path_pt)
        # else:
        #     text = []
        #     dataset = ['train',]
        #     for name in dataset:
                
        #         dataset = name
        #         if name =='dev':
        #             dataset = 'test'
        #         data = []
        #         for rate in ['pos', 'neg']:
        #             input_data_dir = os.path.join(data_path, dataset, rate)
        #             file_list = os.listdir(input_data_dir)
        #             # import pdb; pdb.set_trace()
        #             for file in file_list:
        #                 if file.split('.')[-1]!='txt':
        #                     continue
                        
        #                 with open(os.path.join(input_data_dir,file), 'r') as review:
        #                     text.extend(self.clean_string(review.readline()))
                            
        #     word_frequence = Counter(text)
        #     top_3w = word_frequence.most_common(30000-1) #30000th is the position of unk vector       

        #     for i,word in enumerate(top_3w):
        #         self.itos.append(word)
        #         self.stoi[word] = i

        #     logger.info('Saving vectors to {}'.format(path_pt))
        #     torch.save((self.itos, self.stoi, self.vectors, self.dim), path_pt)
         
    def __len__(self):
        return self.dim       





if __name__ == "__main__":
    start_time = time.time()
    print("Start")
    A = One_hot_vector()
    print("Process Complete in {:3d}s!".format(int(time.time()-start_time)))