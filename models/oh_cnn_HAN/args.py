import os

import models.args
from configparser import ConfigParser

def get_args():
    parser = models.args.get_args()

    parser.add_argument('--mode', type=str, default='static', choices=['rand', 'static', 'non-static'])
    parser.add_argument('--dataset', type=str, default='Reuters')
    parser.add_argument('--output-channel', type=int, default=100)
    parser.add_argument('--words-dim', type=int, default=300)
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch-decay', type=int, default=15)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--word-num-hidden', type=int, default=50)
    parser.add_argument('--sentence-num-hidden', type=int, default=50)

    parser.add_argument('--word-vectors-dir', default=os.path.join(os.pardir, 'hedwig-data', 'embeddings', 'word2vec'))
    parser.add_argument('--word-vectors-file', default='GoogleNews-vectors-negative300.txt')
    parser.add_argument('--save-path', type=str, default=os.path.join('model_checkpoints', 'han'))
    parser.add_argument('--resume-snapshot', type=str)
    parser.add_argument('--trained-model', type=str)

    parser.add_argument('--cfg-name', type=str)

    args = parser.parse_args()
    

    # cfg = ConfigParser()
    # # 对文件修改必须先将文件读取到config
    # cfg.read(args.cfg_name, encoding='UTF-8')
    # cfg.getboolean('model','log_errors')
    

    return args

if __name__ == "__main__":
    args = get_args()