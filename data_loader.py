import dgl
import torch
import pickle
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from IPython import embed
from utils import *


def build_taggnn_graph(args):
    query_wids = pload('dataset/' + args.dataset + '/query_wids')
    item_wids = pload('dataset/' + args.dataset + '/item_wids')
    tag_wids = pload('dataset/' + args.dataset + '/tag_wids')
    vocab_size = 0
    for x in query_wids:
        vocab_size = max(vocab_size,  max(query_wids[x]))
    for x in item_wids:
        vocab_size = max(vocab_size,  max(query_wids[x]))
    for x in tag_wids:
        vocab_size = max(vocab_size,  max(query_wids[x]))
    vocab_size += 1
    args.vocab_size = vocab_size

    n_query, n_item, n_tag = len(query_wids), len(item_wids), len(tag_wids) 
    g = dgl.DGLGraph()
    g.add_nodes(n_query + n_item + n_tag)
    
    if args.model != 'taggnn-it':
        edges1, edges2, edges_feat = [], [], []
        query_item_rels = pload('dataset/' + args.dataset + '/query_item_relations')
        for query in query_item_rels:
            edges1.extend([query - 1] * len(query_item_rels[query]))
            edges2.extend([n_query + item - 1 for item in query_item_rels[query]])
            edges_feat.extend([query_item_rels[query][item] for item in query_item_rels[query]])
        g.add_edges(edges1, edges2)
        g.add_edges(edges2, edges1)
        edges_feat = edges_feat + edges_feat

    # feature
    feats = []
    feats.extend([torch.tensor(query_wids[query]) for query in query_wids])
    feats.extend([torch.tensor(item_wids[item]) for item in item_wids])
    feats.extend([torch.tensor(tag_wids[tag]) for tag in tag_wids])
    
    lengths = torch.tensor([len(feat) for feat in feats]).float()
    feats = pad_sequence(feats, batch_first=True, padding_value=vocab_size)

    g.ndata['len'] = lengths
    g.ndata['wids'] = feats
    g.ndata['idx'] = torch.arange(g.number_of_nodes())
    g.ndata['type'] = torch.tensor([0]*n_query + [1]*n_item + [2]*n_tag)  # node type: 0 query, 1 item, 2 tag
    g.edata['val'] = torch.tensor(edges_feat)
    
    args.n_query = n_query
    args.n_item = n_item
    args.n_tag = n_tag
    
    return args, g



def data_split_and_prepare(args, g):
    # 1.
    # randomly select train_num items as train, half of val_num items for "full tag prediction" 
    # and the remained half of val_num items for tag completion 

    item_list = list(range(args.n_query, args.n_query + args.n_item))
    random.seed(2021)
    random.shuffle(item_list)

    train_items = item_list[:args.train_num]
    val_ftp_items = item_list[args.train_num : args.train_num + int(args.val_num / 2)]   # for full tag prediction
    val_tc_items = item_list[args.train_num + int(args.val_num / 2) : args.train_num + args.val_num] # for tag completion
    test_ftp_items = item_list[args.train_num + args.val_num + int(args.test_num / 2) : \
        args.train_num + args.val_num + args.test_num]
    test_tc_items = item_list[args.train_num + args.val_num + args.test_num : ]


    item_tag_rels = pload('dataset/' + args.dataset + '/item_tag_relations')
    # 2. 
    # if not taggnn-qi, complete the item-tag side's edges
    if args.model != 'taggnn-qi':
        edges1, edges2, edges_feat = [], [], []
        for item in train_items:
            edges1.extend([args.n_query + item - 1] * len(item_tag_rels[item - args.n_query + 1]))
            edges2.extend([tag + args.n_query + args.n_item - 1 for tag in item_tag_rels[item - args.n_query + 1]])
            edges_feat.extend([item_tag_rels[item - args.n_query + 1][tag] for tag in item_tag_rels[item - args.n_query + 1]])

        # the first two tags for evaluation, the other tags for training
        for item in val_tc_items + test_tc_items:
            tag_freq_dict = item_tag_rels[item - args.n_query + 1]
            tags = list(tag_freq_dict.keys())
            for tag in tags[2:]:
                edges1.append(item)
                edges2.append(args.n_query + args.n_item - 1 + tag)
                edges_feat.append(tag_freq_dict[tag])
    
    
    # 3.
    # create label and mask matrices for evaluation
    label_mat = torch.zeros((args.n_item, args.n_tag))
    tc_label_mat = torch.zeros((args.n_item, args.n_tag))
    mask_mat = torch.zeros((args.n_item, args.n_tag))

    for item in train_items:
        tags = torch.tensor(list(item_tag_rels[item - args.n_query + 1].keys())) - 1
        label_mat[item - args.n_query][tags] = 1
    
    for item in val_tc_items + test_tc_items:
        tags = torch.tensor(list(item_tag_rels[item - args.n_query + 1].keys())) - 1
        label_mat[item - args.n_query][tags[2:]] = 1
        mask_mat[item - args.n_query][tags[2:]] = -np.inf

    for item in val_ftp_items + test_ftp_items:
        tags = torch.tensor(list(item_tag_rels[item - args.n_query + 1].keys())) - 1
        label_mat[item - args.n_query][tags] = 1

    for item in val_tc_items + test_tc_items:
        tags = torch.tensor(list(item_tag_rels[item - args.n_query + 1].keys())) - 1
        tc_label_mat[item - args.n_query][tags[:2]] = 1

    return label_mat, tc_label_mat, mask_mat, g, [train_items, val_ftp_items, val_tc_items, test_ftp_items, test_tc_items]

