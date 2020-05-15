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

from datasets.reuters import ReutersHierarchical as Reuters
from datasets.yelp2014 import Yelp2014Hierarchical as Yelp2014
from models.oh_cnn_HAN.args import get_args
from models.oh_cnn_HAN.model import One_hot_CNN
from models.oh_cnn_HAN.one_hot_vector_preprocess import One_hot_vector
from models.oh_cnn_HAN.sentence_tokenize import Sentence_Tokenize

from models.oh_cnn_HAN.xls_writer import write_xls
from models.oh_cnn_HAN.optim_Noam import NoamOpt

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
    
    
    @classmethod
    def unk_oh(cls, tensor):
        size_tup = tuple(tensor.size())
        if size_tup not in cls.cache:
            cls.cache[size_tup] = torch.zeros(tensor.size())
            # cls.cache[size_tup]
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

def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, is_multilabel, is_binary):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    if hasattr(saved_model_evaluator, 'is_multilabel'):
        saved_model_evaluator.is_multilabel = is_multilabel
    if hasattr(saved_model_evaluator, 'ignore_lengths'):
        saved_model_evaluator.ignore_lengths = True
    if hasattr(test_evaluator, 'is_binary'):
        saved_model_evaluator.is_binary = is_binary


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

    # Hyperparameters!
    config = deepcopy(args)
    config.output_channel    = [400, 400]
    config.input_channel     = 30000
    config.kernel_H          = [4, 3]
    config.kernel_W          = [1, 1]
    config.stride            = [1, 1]
    config.rnn_hidden_size   = 400
    config.max_size          = 30000
    config.fill_value        = 1
    config.use_RNN           = True
    config.id                = 1
    config.hierarchical      = True
    config.attention         = True
    config.fix_length        = None
    config.sort_within_batch = True
    config.optimizer_warper  = True
    config.rnn_drop_out      = 0.5

    if config.hierarchical:
        from datasets.imdb_stanford import IMDBHierarchical as IMDB_stanford
        from datasets.imdb import IMDBHierarchical as IMDB
    else:
        from datasets.imdb_stanford import IMDB_stanford 
        from datasets.imdb import IMDB

    dataset_map = {
        'Reuters': Reuters,
        'AAPD': AAPD,
        'IMDB': IMDB,
        'Yelp2014': Yelp2014,
        'IMDB_2':IMDB_2,
        'IMDB_stanford':IMDB_stanford,
    }

    dataset_map['IMDB'].NESTING_FIELD = Field(batch_first=True, tokenize=clean_string,  fix_length = config.fix_length )
    dataset_map['IMDB'].TEXT_FIELD = NestedField(dataset_map['IMDB'].NESTING_FIELD, tokenize=Sentence_Tokenize())
    # '''Notetice that these just a place holder. Thus, vector attributes of field is Null'''
    # one_hot_vector = One_hot_vector()
    
    time_tmp = time.time()

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')
    else:
        dataset_class = dataset_map[args.dataset]
        train_iter, dev_iter, test_iter = dataset_class.iters(args.data_dir,
                                                            #   args.word_vectors_file,
                                                            #   args.word_vectors_dir,
                                                                batch_size=args.batch_size,
                                                                device=args.gpu,
                                                            #   unk_init=UnknownWordVecCache.unk_oh,  
                                                            #   vectors = one_hot_vector
                                                                onehot_Flag =True,
                                                                max_size = config.max_size,
                                                                sort_within_batch = config.sort_within_batch
                                                              )

    config.dataset = train_iter.dataset
    config.target_class = train_iter.dataset.NUM_CLASSES
    config.words_num = len(train_iter.dataset.TEXT_FIELD.vocab)
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
        model = One_hot_CNN(config)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = torch.nn.DataParallel(model)
        if args.cuda:
            model.cuda()

    if not args.trained_model:
        save_path = os.path.join(args.save_path, dataset_map[args.dataset].NAME)
        os.makedirs(save_path, exist_ok=True)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    

    # optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay,  betas=(0.9, 0.98), eps=1e-9)
    
    config.ow_factor = 4
    config.ow_warmup = 80000
    config.ow_model_size = 600
    if config.optimizer_warper:
        optimizer = NoamOpt( config.ow_model_size, config.ow_factor, config.ow_warmup, optimizer)
    # optimizer = torch.optim.RMSprop(parameter, lr=args.lr, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0.9, centered=False)
    # optimizer = torch.optim.SGD(parameter, lr=args.lr, momentum=0.9)
    
    train_evaluator = EvaluatorFactory.get_evaluator(dataset_class, model, None, train_iter, args.batch_size, args.gpu)
    test_evaluator  = EvaluatorFactory.get_evaluator(dataset_class, model, None, test_iter, args.batch_size, args.gpu)
    dev_evaluator   = EvaluatorFactory.get_evaluator(dataset_class, model, None, dev_iter, args.batch_size, args.gpu)

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

    args.patience =5
    is_binary = True if config.target_class == 2 else False
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
        'optimizer_warper': config.optimizer_warper
    }

    trainer = TrainerFactory.get_trainer(args.dataset, model, None, train_iter, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.trained_model:
        dev_results, train_result = trainer.train(args.epochs)
    else:
        if args.cuda:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))
        else:
            model = torch.load(args.trained_model, map_location=lambda storage, location: storage)

    write_xls(train_result, dev_results, config)



    # Calculate dev and test metrics
    if hasattr(trainer, 'snapshot_path'):
        model = torch.load(trainer.snapshot_path)

    evaluate_dataset('dev', dataset_map[args.dataset], model, None, dev_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu,
                     is_binary = is_binary
                     )
    evaluate_dataset('test', dataset_map[args.dataset], model, None, test_iter, args.batch_size,
                     is_multilabel=dataset_class.IS_MULTILABEL,
                     device=args.gpu,
                     is_binary = is_binary
                     )
