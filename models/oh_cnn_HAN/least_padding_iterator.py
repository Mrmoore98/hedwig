import math
import random

import torch

from torchtext.data.utils import RandomShuffler
from torchtext.data.batch import Batch
from torchtext.data.dataset import Dataset
from torchtext.data.iterator import Iterator

class Less_padding_bucket_Iterator(Iterator):
    """Defines an iterator that batches examples of similar lengths together.
    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """
    def __init__(self, dataset, batch_size, sort_key=None, device=None, batch_size_fn=None, 
                       train=True, repeat=None, shuffle=None, sort=None, sort_within_batch=None, bucket_size=1000):
        super(Less_padding_bucket_Iterator, self).__init__(
                    dataset =dataset, 
                    batch_size = batch_size, 
                    sort_key = sort_key, 
                    device=device,
                    batch_size_fn=batch_size_fn, 
                    train=train,
                    repeat=repeat, 
                    shuffle=shuffle, 
                    sort=sort,
                    sort_within_batch=sort_within_batch)
        self.bucket_size = bucket_size

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key, self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch,
                                bucket_size= self.bucket_size)


def batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch


def pool(data, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
        random_shuffler=None, shuffle=False, sort_within_batch=False, bucket_size= 1000):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle
    for p in batch(data, batch_size * 100, batch_size_fn):
        p_batch = batch(sorted(p, key=key), batch_size, batch_size_fn) \
            if sort_within_batch \
            else batch(p, batch_size, batch_size_fn)
        if shuffle:
            for b in random_shuffler(list(p_batch)):
                yield b
        else:
            for b in list(p_batch):
                yield b