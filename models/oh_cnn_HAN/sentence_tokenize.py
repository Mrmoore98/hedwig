
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
        self.threshold = 25


    def split_sentence(self,string):
    
        string = re.sub(r"<br />", " ", string) # get rid of huanhangfu
        # string = re.sub(r"[^A-Za-z0-9():.,!?\'`]", " ", string)   
        string = re.sub(r"[!?]"," ", string)
        
        # import pdb; pdb.set_trace()
        if self.method =='origin':
            sentence_list = self.st(string)
        elif self.method == 'uslearn':
            sentence_list = self.pst.sentences_from_text(string)
        sentence_num = len(sentence_list)
        if sentence_num > self.threshold:   
            step = len(string)//self.threshold
            sentence_list = [string[i:i+step] for i in range(0, len(string), step)]

        return sentence_list 

    def __call__(self, string):
        return self.split_sentence(string)