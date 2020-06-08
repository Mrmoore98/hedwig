import _pickle as cPickle
import numpy as np


def Seq_Split(seq, split_list, pad_zero_index):

    seq_array = np.array(seq)
    split_indexes = []
    for split_symbol in split_list:
        tmp = np.where(seq_array == split_symbol)
        split_indexes.extend(tmp[0].tolist())
    split_indexes.sort()

    result = []
    result_len = []
    if len(split_indexes) == 0:
        seq = [pad_zero_index] + seq + [pad_zero_index]
        result.append(seq)
        result_len.append(len(seq))
    else:
        split_start = 0
        for split_index in split_indexes:
            sub_seq = seq[split_start:split_index+1]
            sub_seq = [pad_zero_index] + sub_seq + [pad_zero_index]
            result.append(sub_seq)
            result_len.append(len(sub_seq))
            split_start = split_index+1

        if split_start != len(seq):
            sub_seq = seq[split_start:]
            sub_seq = [pad_zero_index] + sub_seq + [pad_zero_index]
            result.append(sub_seq)
            result_len.append(len(sub_seq))

    return [result_len, result]

def Seq_Max_Len(seqs):

    max_len = 0
    for i in range(len(seqs)):
        if len(seqs[i]) > max_len:
            max_len = len(seqs[i])
    return max_len

def Seq_Min_Len(seqs):

    min_len = 999
    for i in range(len(seqs)):
        if len(seqs[i]) < min_len:
            min_len = len(seqs[i])
    return min_len

def Seq_Ave_Len(seqs):
    ave_len = 0
    for i in range(len(seqs)):
        if i == 0:
            ave_len = ave_len + len(seqs[i])
        else:
            ave_len = (ave_len + len(seqs[i]))/2
    return ave_len

def Word_Max_len(words):
    max_len = 0
    ww = [0] * len(words)
    for i in range(len(words)):
        for j in range(len(words[i])):
            ww[i] = ww[i] + len(words[i][j])
    for i in range(len(ww)):
        if ww[i] > max_len:
            max_len = ww[i]
    return max_len



def Load_Data(data_name):





    if data_name == 'IMDB':
        data = cPickle.load(open('./imdb.pkl', 'rb'), encoding='iso-8859-1')
    if data_name == 'ELEC':
        data = cPickle.load(open('./ELEC_30k_cwc.pkl', 'rb'),encoding='iso-8859-1')
    if data_name == 'IMDB_10':
        new_data = cPickle.load(open('/home/s/CNN-BiLSTM2/hedwig/BCPGDS_decoder/imdb_data.pkl', 'rb'))
    if data_name == 'Reuters':
        new_data = cPickle.load(open('./Reuters_data.pkl', 'rb'))
        new_data_bow = cPickle.load(open('./Reuters_data_bow.pkl', 'rb'))


    if data_name == 'IMDB_10' or data_name == 'Reuters':
        #doc_labels = data['labels']
        #word_freq = data['word_freq']
        word2index = new_data.stoi
        #word2index = {key: idx for key, idx in word2index1.items() if idx < 3000}
        index2word =  new_data.itos
        #train_doc_word = data['train_doc_word']
        train_doc_split = new_data.data['train']
        if data_name == 'IMDB_10':
            train_doc_split = train_doc_split
            train_doc_label = np.array(new_data.label['train'])
        else:
            train_doc_split = train_doc_split
            train_doc_label = np.array(new_data_bow.label['train'])
        if data_name == 'IMDB_10':
            train_doc_label = train_doc_label
        else:
            train_doc_label = train_doc_label
        #test_doc_word = data['test_doc_word']
        test_doc_split = new_data.data['test']
        #test_doc_label = np.array(new_data_bow.label['test'])
        if data_name == 'IMDB_10':
            test_doc_split = test_doc_split
            test_doc_label = np.array(new_data.label['test'])
        else:
            test_doc_split = test_doc_split
            test_doc_label = np.array(new_data_bow.label['test'])
        seq_max_len = 0
        # seq_min_len = 999
        train_doc_len = []
        for i in range(len(train_doc_split)):
            train_doc_len.append(len(train_doc_split[i]))
        test_doc_len = []
        for i in range(len(test_doc_split)):
            test_doc_len.append(len(test_doc_split[i]))

        Data_save = {}
        Data_save['word2index'] = word2index
        Data_save['index2word'] = index2word
        Data_save['train_doc_split'] = train_doc_split
        Data_save['train_doc_label'] = train_doc_label
        Data_save['test_doc_split'] = test_doc_split
        Data_save['test_doc_label'] = test_doc_label
        cPickle.dump(Data_save, open('./Reuters_new.pkl', 'wb'))

    else:

        if data_name == 'ELEC':
            doc_labels = data['Label']
        else:
            doc_labels = data['labels']
        word_freq = data['word_freq']
        word2index = data['word2index']
        index2word = data['index2word']
        train_doc_word = data['train_doc_word']
        train_doc_index = data['train_doc_index']
        train_doc_label = np.array(data['train_doc_label'])
        test_doc_word = data['test_doc_word']
        test_doc_index = data['test_doc_index']
        test_doc_label = np.array(data['test_doc_label'])

        #==================================================
        #preprocess

        num_words = len(index2word)
        index2word[num_words] = '<pad_zero>'
        word2index['<pad_zero>'] = num_words
        num_words = num_words + 1

        # num_words = len(index2word)
        # index2word[1] = '<pad_zero>'
        # word2index['<pad_zero>'] = 1
        # #num_words = num_words + 1

        seq_max_len = 0
        # seq_min_len = 999
        train_doc_split = []
        train_doc_split_len = []
        train_doc_len = []
        split_index = [word2index['.'], word2index['!'], word2index['?'], word2index['..'], word2index[';']]

        for i in range(len(train_doc_index)):

            [seqs_len, seqs] = Seq_Split(train_doc_index[i], split_index, word2index['<pad_zero>'])
            train_doc_split.append(seqs)
            train_doc_split_len.append(seqs_len)

            # tmp_min = Seq_Min_Len(seqs)
            # if tmp_min < seq_min_len:
            #     seq_min_len = tmp_min
            tmp_max = Seq_Max_Len(seqs)
            if tmp_max > seq_max_len:
                seq_max_len = tmp_max

            train_doc_len.append(len(seqs))

        test_doc_split = []
        test_doc_split_len = []
        test_doc_len = []

        for i in range(len(test_doc_index)):

            [seqs_len, seqs] = Seq_Split(test_doc_index[i], split_index, word2index['<pad_zero>'])
            test_doc_split.append(seqs)
            test_doc_split_len.append(seqs_len)

            # tmp_min = Seq_Min_Len(seqs)
            # if tmp_min < seq_min_len:
            #     seq_min_len = tmp_min
            tmp_max = Seq_Max_Len(seqs)
            if tmp_max > seq_max_len:
                seq_max_len = tmp_max

            test_doc_len.append(len(seqs))

    doc_max_len = max(Seq_Max_Len(train_doc_split), Seq_Max_Len(test_doc_split))
    doc_min_len = min(Seq_Min_Len(train_doc_split), Seq_Min_Len(test_doc_split))
    doc_max_len_word = max(Word_Max_len(train_doc_split), Word_Max_len(test_doc_split))
    doc_ave_len = (Seq_Ave_Len(train_doc_split) + Seq_Ave_Len(test_doc_split))/2

    return word2index, train_doc_split, train_doc_label, train_doc_len, test_doc_split, test_doc_label, test_doc_len
