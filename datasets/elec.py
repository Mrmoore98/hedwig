import os

import numpy as np
import torch
from torchtext.data import NestedField, Field, TabularDataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from datasets.reuters import clean_string, split_sents, process_labels, generate_ngrams
from models.oh_cnn_HAN.sentence_tokenize import Sentence_Tokenize
from models.oh_cnn_HAN.least_padding_iterator import Less_padding_bucket_Iterator

def char_quantize(string, max_length=500):
    identity = np.identity(len(ELECCharQuantized.ALPHABET))
    quantized_string = np.array([identity[ELECCharQuantized.ALPHABET[char]] for char in list(string.lower()) if char in ELECCharQuantized.ALPHABET], dtype=np.float32)
    if len(quantized_string) > max_length:
        return quantized_string[:max_length]
    else:
        return np.concatenate((quantized_string, np.zeros((max_length - len(quantized_string), len(ELECCharQuantized.ALPHABET)), dtype=np.float32)))


class ELEC(TabularDataset):
    NAME = 'ELEC'
    NUM_CLASSES = 2
    IS_MULTILABEL = False

    TEXT_FIELD  = Field(batch_first=True, tokenize=clean_string, include_lengths=True)
    LABEL_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=process_labels)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, path, train=os.path.join('ELEC', 'train.tsv'),
               validation=os.path.join('ELEC', 'dev.tsv'),
               test=os.path.join('ELEC', 'test.tsv'), **kwargs):
        return super(ELEC, cls).splits(
            path, train=train, validation=validation, test=test,
            format='tsv', fields=[('label', cls.LABEL_FIELD), ('text', cls.TEXT_FIELD)]
        )

    @classmethod
    def iters(cls, path, vectors_name=None, vectors_cache=None, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_, onehot_Flag =False, max_size = None,  sort_within_batch=False, bucket_size=300):
        """
        :param path: directory containing train, test, dev files
        :param vectors_name: name of word vectors file
        :param vectors_cache: path to directory containing word vectors file
        :param batch_size: batch size
        :param device: GPU device
        :param vectors: custom vectors - either predefined torchtext vectors or your own custom Vector classes
        :param unk_init: function used to generate vector for OOV words
        :return:
        """
        if vectors is None and not onehot_Flag:
            vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=unk_init)
        if max_size is not None: max_size = max_size-2
        train, val, test = cls.splits(path)
        cls.TEXT_FIELD.build_vocab(train, val, test, vectors=vectors, max_size= max_size)
        return Less_padding_bucket_Iterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle,
                                     sort_within_batch=sort_within_batch, device=device, bucket_size=bucket_size)


class ELECHierarchical(ELEC):
    NESTING_FIELD = Field(batch_first=True, tokenize=clean_string,  fix_length = 50 )
    TEXT_FIELD = NestedField(NESTING_FIELD, tokenize=Sentence_Tokenize())


class ELECCharQuantized(ELEC):
    ALPHABET = dict(map(lambda t: (t[1], t[0]), enumerate(list("""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"""))))
    TEXT_FIELD = Field(sequential=False, use_vocab=False, batch_first=True, preprocessing=char_quantize)

    @classmethod
    def iters(cls, path, vectors_name, vectors_cache, batch_size=64, shuffle=True, device=0, vectors=None,
              unk_init=torch.Tensor.zero_):
        """
        :param path: directory containing train, test, dev files
        :param batch_size: batch size
        :param device: GPU device
        :return:
        """
        train, val, test = cls.splits(path)
        return BucketIterator.splits((train, val, test), batch_size=batch_size, repeat=False, shuffle=shuffle, device=device)
