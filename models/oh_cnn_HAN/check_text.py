import csv
import os 
import spacy
import re
import time
import numpy as np
import random
# spacy=spacy.load('en')


data_path = '/home/zhangxin/sentiment_analysis/hedwig-data/datasets/aclImdb/' 
output_data_path = '/home/zhangxin/sentiment_analysis/hedwig-data/datasets/IMDB_stanford/' 




def match_str(input, itos, threshold = 20):

    threshold = 1000
    # only find contain words
    input = input.cpu().numpy()
  


    def search(idx_input, text, itos = itos):
        
        match_count = 0
        for word_id in idx_input:
            if  itos[word_id] in text:
                match_count += 1

        return match_count  

    def do_search(input, threshold):
        input = np.unique(input)
        for name in ['train']:
            
            dataset = name
            data    = []
            index   = []
            for rate in ['pos', 'neg']:

                input_data_dir = os.path.join(data_path, dataset, rate)
                file_list = os.listdir(input_data_dir)
                # import pdb; pdb.set_trace()
                for file in file_list:
                    if file.split('.')[-1]!='txt':
                        continue
                    
                    with open(os.path.join(input_data_dir,file), 'r') as review:
                        text = review.readline()
                        if search(input ,text) >= len(input):
                            data.append(text)
                            index.append(file)
        
        return data, index
    
    doc = []
    for doc_i in input:
        for sent_i in doc_i:

            sentence = [itos[idx] for idx in sent_i]
            sentence = list(filter(lambda t: t!='<pad>', sentence))
            doc.append(sentence)
            print(sentence)
        
        import pdb; pdb.set_trace()
        data, index = do_search(doc_i.reshape(-1),100)
        

    return 0




def match_str(input, itos, threshold = 20):

    threshold = 1000
    # only find contain words
    input = input.cpu().numpy()
  


    def search(idx_input, text, itos = itos):
        
        match_count = 0
        for word_id in idx_input:
            if  itos[word_id] in text:
                match_count += 1

        return match_count  

    def do_search(input, threshold):

        input = np.unique(input)
        data    = []
        index   = []
        most_similar_count = 0
        for name in ['train']:
            
            dataset = name
            
            for rate in ['pos', 'neg']:

                input_data_dir = os.path.join(data_path, dataset, rate)
                file_list = os.listdir(input_data_dir)
                # import pdb; pdb.set_trace()
                for file in file_list:
                    if file.split('.')[-1]!='txt':
                        continue
                    
                    with open(os.path.join(input_data_dir,file), 'r') as review:
                        text = review.readline()
                        new_count = search(input ,text)
                        if new_count > most_similar_count:
                            
                            most_similar_count = new_count
                            data = text
                            index= file

        
        return data, index
    
    doc = ''
    for doc_i in input:
        for sent_i in doc_i:

            sentence = [itos[idx] for idx in sent_i]
            sentence = list(filter(lambda t: t!='<pad>', sentence))
            sentence = ' '.join(sentence)
            doc      = doc + sentence + ' ' 
        

        print(doc)
        
        import pdb; pdb.set_trace()
        data, index = do_search(doc_i.reshape(-1),100)
        

    return 0









if __name__ == "__main__":
    start_time = time.time()
    print("Start")
    merge_doc()
    print("Process Complete in {:3d}s!".format(int(time.time()-start_time)))