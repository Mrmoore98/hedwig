# coding=utf-8

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

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import reuters
from nltk.tokenize import TweetTokenizer, sent_tokenize

import pickle
from xlwt import *


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
        # self.vector_init()
        self.pst = PunktSentenceTokenizer()
        # self.pre_train()
        self.st  = sent_tokenize
        self.method = 'origin'
        self.threshold = 25
       
    def clean_string(self, string):
        """
        Performs tokenization and string cleaning for the Reuters dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.lower().strip().split()
    
    def clean_string_np(self, string):
        """
        Performs tokenization and string cleaning for the Reuters dataset
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = string.lower().strip().split()
        res    = np.array(list(map(self.stoi_take, string )))
        
        return res
    
    def stoi_take(self, word):
        if word in self.stoi.keys():
            return self.stoi[word]
        else: 
            return 0


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
  
    def split_sentence(self, string):
        
        string = re.sub(r"<br />", " ", string) # get rid of huanhangfu
        # string = re.sub(r"[^A-Za-z0-9():.,!?\'`]", " ", string)   
        string = re.sub(r"[!?]"," ", string)
        
        # import pdb; pdb.set_trace()
        if self.method =='origin':
            sentence_list = self.st(string)
        elif self.method == 'uslearn':
            sentence_list = self.pst.sentences_from_text(string)
        sentence_num = len(sentence_list)
        # if sentence_num > self.threshold:   
        #     step = len(string)//self.threshold
        #     sentence_list = [string[i:i+step] for i in range(0, len(string), step)]

        return sentence_list 

    def count_word(self, string):
        string = re.sub(r"[^A-Za-z0-9():.,!?\'`]", " ", string) 
        return len(string.strip().split())

    def split_doc(self):

        data_path = '/home/zhangxin/sentiment_analysis/hedwig-data/datasets/aclImdb/' 
        output_data_path = '/home/zhangxin/sentiment_analysis/hedwig-data/datasets/IMDB_stanford/' 

    
    
        text = []
        max_sentence = 0
        sentence_len = []
        max_word_len = []
        dataset = ['train','test']
        for name in dataset:
            
            dataset = name
            if name =='dev':
                dataset = 'test'
            data = []
            for rate in ['pos', 'neg']:
                input_data_dir = os.path.join(data_path, dataset, rate)
                file_list = os.listdir(input_data_dir)
                # import pdb; pdb.set_trace()
                for file in file_list:
                    if file.split('.')[-1]!='txt':
                        continue
                    
                    with open(os.path.join(input_data_dir,file), 'r') as review:
                        sen_list = self.split_sentence(review.readline()) 
                        max_sentence = max(len(sen_list), max_sentence)
                        sentence_len.append(len(sen_list))
                        max_word_len.append(max([self.count_word(sen) for sen in sen_list]))
                        # if len(sen_list)>117:
                            # import pdb; pdb.set_trace()

        sentence_len.sort()       
        max_word_len.sort()         
        print(max_sentence)  
        self.wirte_xls([sentence_len,max_word_len])                      
        # import pdb; pdb.set_trace()

        logger.info('Success!')
    
    def pre_train(self):
        self.pst.train(reuters.raw())
        # trainer = PunktTrainer()
        # trainer.INCLUDE_ALL_COLLOCS = True
        # trainer.train(text)
        # tokenizer = PunktSentenceTokenizer(trainer.get_params())
        
        logger.info('pretrain success!')

         
    def __len__(self):
        return self.dim       

    def wirte_xls(self, data):
     
        file = Workbook(encoding = 'utf-8')
        #指定file以utf-8的格式打开
        table = file.add_sheet('data')
        #指定打开的文件名

        method = 'origin'
        num_freq = Counter(data[0])
        word_num_freq = Counter(data[1])
        for i,num in enumerate(data[0]):
            table.write(i,1, num)
        
        for i, name in enumerate(num_freq):
            table.write(i,2, name)
            table.write(i,3, num_freq[name])
           
        for i, name in enumerate(word_num_freq):
            table.write(i,4, name)   
            table.write(i,5, word_num_freq[name])

        file.save('/home/zhangxin/sentiment_analysis/hedwig/models/oh_cnn_HAN/data_{}.xls'.format(method))

    def onehot2int(self, label):
        '''str -> int'''

        label = list(label)        
        return label.index('1')

    def tsv2np(self):
        
        dataset = ['train', 'test']
        output_file = './imdb_data.pkl'
        data_path   = '/home/s/CNN-BiLSTM2/hedwig-data/datasets/IMDB/' 
        data = {}
        labels= {}
        for name in dataset:
            doc = []
            doc_count = []
            labels[name] = []
            with open(data_path + '{}.tsv'.format(name), 'r') as input_tsv:
                
                tsv_data_raw = csv.reader(input_tsv, delimiter='\t')
                for label, text in tsv_data_raw:
                    
                    labels[name].append(self.onehot2int(label))

                    if name == 'train':
                        doc_count.extend(self.clean_string(text))
                
             
                if name == 'train':                
                    word_frequence = Counter(doc_count)
                    top_3w = word_frequence.most_common(30000-2) #30000th is the position of unk vector       
                    self.itos.extend(['<UNK>','<PAD>'])
                    self.stoi['<UNK>'] = 0
                    self.stoi['<PAD>'] = 1

                    for i,(word, count) in enumerate(top_3w):
                        self.itos.append(word)
                        self.stoi[word] = i + 2

            with open(data_path + '{}.tsv'.format(name), 'r') as input_tsv:

                tsv_data_raw = csv.reader(input_tsv, delimiter='\t')
                for label, text in tsv_data_raw:
                    
                    doc_tmp = list(map(self.clean_string_np, self.split_sentence(text)))
                    doc.append(doc_tmp)

            data[name] = doc
  
        data_store = [self.itos,self.stoi,labels,data]
        with open(output_file,'wb') as file:
            pickle.dump(data_store, file)
        
        
        


                    
                    

            
class IMDB_data_struct(object):

    def __init__(self,itos,stoi,label,data):
        self.itos = itos
        self.stoi = stoi
        self.label = label            
        self.data  = data




if __name__ == "__main__":
    start_time = time.time()
    print("Start")
    A = One_hot_vector()
    A.tsv2np()
    print("Process Complete in {:3d}s!".format(int(time.time()-start_time)))
    output_file = './imdb_data.pkl'
    with open(output_file,'rb') as file:
            res = pickle.load(file)
    import pdb; pdb.set_trace()