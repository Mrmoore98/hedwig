import logging
import os
import random
from copy import deepcopy
import time

import numpy as np
import torch
import torch.onnx
from torchtext.data import NestedField, Field, TabularDataset
from datasets.reuters import clean_string, split_sents, process_labels, generate_ngrams

from common.evaluate import EvaluatorFactory
from common.train import TrainerFactory
from datasets.aapd import AAPDHierarchical as AAPD
from datasets.imdb import IMDBHierarchical as IMDB
from datasets.imdb_2 import IMDBHierarchical as IMDB_2
from datasets.imdb_stanford import IMDBHierarchical as IMDB_stanford
from datasets.elec import ELECHierarchical as ELEC
from datasets.reuters import ReutersHierarchical as Reuters
from datasets.yelp2014 import Yelp2014Hierarchical as Yelp2014
from models.han.args import get_args
from models.han.model import HAN

from models.oh_cnn_HAN.xls_writer import write_xls
from models.oh_cnn_HAN.optim_Noam import NoamOpt
from models.oh_cnn_HAN.sentence_tokenize import Sentence_Tokenize, Word_Tokenize

class UnknownWordVecCache(object):
    """
    Caches the first randomly generated word vector for a certain size to make it is reused.
    """
    cache = {}

    @classmethod
    def unk(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.Tensor(tensor.size())
            cls.cache[size_tup].uniform_(-0.25, 0.25)
        return cls.cache[size_tup]


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, is_multilabel):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    if hasattr(saved_model_evaluator, 'is_multilabel'):
        saved_model_evaluator.is_multilabel = is_multilabel
    if hasattr(saved_model_evaluator, 'ignore_lengths'):
        saved_model_evaluator.ignore_lengths = True

    scores, metric_names = saved_model_evaluator.get_scores()
    print('Evaluation metrics for', split_name)
    print(metric_names)
    print(scores)



if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    logger = get_logger()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not args.cuda:
        args.gpu = -1
    if torch.cuda.is_available() and args.cuda:
        print('Note: You are using GPU for training')
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.cuda:
        print('Warning: Using CPU for training')

    dataset_map = {
        'Reuters': Reuters,
        'AAPD': AAPD,
        'IMDB': IMDB,
        'Yelp2014': Yelp2014,
        'IMDB_2':IMDB_2,
        'IMDB_stanford':IMDB_stanford,
        'ELEC':ELEC,
    }
    config = deepcopy(args)
    config.fix_length = None
    config.sort_within_batch = True

    dataset_map[args.dataset].NESTING_FIELD = Field(batch_first=True, tokenize=Word_Tokenize(),  fix_length = config.fix_length )
    dataset_map[args.dataset].TEXT_FIELD = NestedField(dataset_map[args.dataset].NESTING_FIELD, tokenize=Sentence_Tokenize())

    time_tmp = time.time()
    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    else:
        dataset_class = dataset_map[args.dataset]
        train_iter, dev_iter, test_iter = dataset_class.iters(args.data_dir,
                                                              args.word_vectors_file,
                                                              args.word_vectors_dir,
                                                              batch_size=args.batch_size,
                                                              device=args.gpu,
                                                              unk_init=UnknownWordVecCache.unk,
                                                              sort_within_batch = config.sort_within_batch
                                                              )

    
    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)

    config.residual = False
    config.cnn = False
    config.dropout_rate = 0.5
    config.optimizer_warper = True
    config.kernel_set = [1,2,3,4]
    config.word_num_hidden = 100
    config.sentence_num_hidden = 50
    config.weight_decay = 1e-5
    
    config.loss = None
    #label smoothing    
    config.label_smoothing = True
    config.std = 0.4
    config.smoothing = 0.05
    config.ls_mode = 'origin'
    # for vae
    config.vae_struct = False

    #front end cnn
    config.frontend_cnn = False


    is_binary = True if config.target_class == 2 else False
    config.is_binary = is_binary
    print('Finished preprocess data in {:.0f}s'.format(time.time()-time_tmp))
    print('Dataset:', args.dataset)
    print('No. of target classes:', train_iter.dataset.NUM_CLASSES)
    print('No. of train instances', len(train_iter.dataset))
    print('No. of dev instances', len(dev_iter.dataset))
    print('No. of test instances', len(test_iter.dataset))

    if args.resume_snapshot:
        if args.cuda:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage)
    else:
        model = HAN(config)
        if args.cuda:
            model.cuda()

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    
    # optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay,  betas=(0.9, 0.98), eps=1e-9)
    optimizer = torch.optim.AdamW(parameter, lr=args.lr, betas=(0.9, 0.98), eps=1e-09, weight_decay=args.weight_decay, amsgrad=True)
    config.ow_factor = 2
    config.ow_warmup = 20000
    config.ow_model_size = 300
    if config.optimizer_warper:
        optimizer = NoamOpt( config.ow_model_size, config.ow_factor, config.ow_warmup, optimizer)
   
    
    train_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, train_iter, args.batch_size, args.gpu)
    test_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, test_iter, args.batch_size, args.gpu)
    dev_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, dev_iter, args.batch_size, args.gpu)

    if hasattr(train_evaluator, 'is_multilabel'):
        train_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(dev_evaluator, 'is_multilabel'):
        dev_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(dev_evaluator, 'ignore_lengths'):
        dev_evaluator.ignore_lengths = True
    if hasattr(test_evaluator, 'is_multilabel'):
        test_evaluator.is_multilabel = dataset_class.IS_MULTILABEL
    if hasattr(test_evaluator, 'ignore_lengths'):
        test_evaluator.ignore_lengths = True

    if hasattr(dev_evaluator, 'is_binary'):
        dev_evaluator.is_binary = is_binary
    if hasattr(test_evaluator, 'is_binary'):
        test_evaluator.is_binary = is_binary

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_every,
        'patience': args.patience,
        'model_outfile': args.save_path,
        'logger': logger,
        'is_multilabel': dataset_class.IS_MULTILABEL,
        'ignore_lengths': True,
        'Binary': is_binary,
        'optimizer_warper': config.optimizer_warper,
        'loss': config.loss
    }

    trainer = TrainerFactory.get_trainer(args.dataset, model, None, train_iter, trainer_config, train_evaluator, test_evaluator, dev_evaluator, config)

    if not args.trained_model:
        dev_results, train_result = trainer.train(args.epochs)
        write_xls(train_result, dev_results, config)
    else:
        if args.cuda:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

    
    # Calculate dev and test metrics
    if hasattr(trainer, 'snapshot_path'):
        model = torch.load(trainer.snapshot_path)

    evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu)
    evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu)
