
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import reuters
from nltk.tokenize import TweetTokenizer, sent_tokenize
import re


class Sentence_Tokenize(object):

    def __init__(self):

        self.pst = PunktSentenceTokenizer()
        # self.pre_train()
        self.st  = sent_tokenize
        self.method = 'origin'
        self.threshold = 15


    def split_sentence(self,string):
    
        string = re.sub(r"<br />", " ", string) # get rid of huanhangfu
        string = re.sub(r"[^A-Za-z0-9():.,!?\'`]", " ", string)   
        string = re.sub(r"([.?!](\s*)){2,}",".", string) 
        sentence_list = re.split(r'[.!?]',string.strip())
        sentence_list = list(filter(None, sentence_list))

        return sentence_list 

    def __call__(self, string):
        return self.split_sentence(string)


class Word_Tokenize(object):

    def __init__(self):
        pass
       


    def split_word(self,string):
     
        # import pdb; pdb.set_trace()
        string = re.sub(r"\s{2,}", " ", string)
        return string.lower().strip().split()


    def __call__(self, string):
        return self.split_word(string)