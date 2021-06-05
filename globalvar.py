import argparse
import os
from os.path import join as oj
import torch
from utils import *
from pprint import pprint
from IPython import embed

def get_params():
    parser = argparse.ArgumentParser(description='TagGNN')
    parser.add_argument('--dataset', type = str,  help='dataset name')
    # parser.add_argument('--train_batch_size', type = int, default=1024,  help='train_batch_size')
    # parser.add_argument('--val_batch_size', type = int, default=4096, help='val_batch_size')
    # parser.add_argument('--test_batch_size', type = int, default=4096,  help='test_batch_size')
    
    parser.add_argument('--train_num', type = int, default=4096,  help='number of train items')
    parser.add_argument('--test_num', type = int, default=4096,  help='number of test items')
    parser.add_argument('--val_num', type = int, default=4096,  help='number of validation items')
    # parser.add_argument('--label_num', type = int, default=4096,  help='number of validation items')
    parser.add_argument('--embedding_dim', type = int, default=100,  help='node and word embedding dimension')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='dropout rate for GNN')
    parser.add_argument('--initial_weight', type=float, default=0.01, help='initial weight for embeddings')

    parser.add_argument('--lr', type = float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type = float, default=0, help='l2 norm')
    parser.add_argument('--model', type = str, default='taggnn', choices=['taggnn-it', 'taggnn-qi', 'taggnn'], help='model name')
    parser.add_argument('--max_epoch', type = int, default = 500, help='max training epoch')
    parser.add_argument('--topk', type = str, default='[1,3,5]', help='evaluation topk')
    parser.add_argument('--val_base_metric', type = str, default='recall', help='based on which metric for early stopping, e.g., p@10')
    parser.add_argument('--early_stop_epoch', type = int, default=10, help='if the val metric does not increase in successive 10 epochs, the training will be stopped')

    parser.add_argument('--only_eval', type = int, default=0, help='whether only evaluate the existing model.')
    parser.add_argument('--eval_model_path', type = str, default=None, help='the load path of the trained model, if only_eval')

    args, _ = parser.parse_known_args()

    return args



args = get_params()
# other basic info
device = auto_gpu_setting()
args.device = device

args.topk = eval(args.topk)
args.topk.sort()

cur_time = now()
print('This Experiment started at {}'.format(cur_time))
print('Arguments Settings:')
pprint(args.__dict__)


# loading and preprocessing data

