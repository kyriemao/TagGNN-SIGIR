from IPython import embed
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from os.path import join as oj
from data_loader import build_taggnn_graph, data_split_and_prepare
from models import TagGNN_QI
from globalvar import *



def train_model(args, g, label_mat, tc_label_mat, mask_mat, all_train_items, val_ftp_items, val_tc_items):
    if args.model == 'taggnn-qi':
        model = TagGNN_QI(args)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    model = model.to(args.device)
    g = g.to(args.device)   
    label_mat = label_mat.to(args.device)
    tc_label_mat = tc_label_mat.to(args.device)

    for epoch in range(args.max_epoch):
        model.train()

        optimizer.zero_grad()
        loss = model.cal_loss(g, label_mat, all_train_items)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            res_metrics = evaluate_model(args, model, g, label_mat, tc_label_mat, mask_mat, all_train_items, val_ftp_items, val_tc_items)
            print('Epoch = {}, Loss = {}'.format(epoch, loss.item()))
            print(','.join([metric + '=' + str(res_metrics[metric]) for metric in res_metrics]))
        
    return  model



def evaluate_model(args, model, g, label_mat, tc_label_mat, mask_mat, all_train_items, ftp_items, tc_items):
    model.eval()

    with torch.no_grad():
        logits = model.test_forward(g).cpu()
        logits += mask_mat
        
        pred_mat = torch.zeros(logits.size()).long()
        res_idx = torch.topk(logits[ftp_items - args.n_query], max(args.topk))[1]
        for i, x in enumerate(res_idx):
            pred_mat[i][x] = 1
        res_idx = torch.topk(logits[tc_items - args.n_query], max(args.topk))[1]
        for i, x in enumerate(res_idx):
            pred_mat[i][x] = 1

        embed()

        label_mat1 = label_mat[ftp_items - args.n_query].long().cpu().numpy()
        res_mat = pred_mat[ftp_items - args.n_query].numpy() & label_mat1

        # ftp_recall = (res_mat.sum(1) / label_mat.sum(1)).mean()
        ftp_precision = (res_mat.sum(1) / max(args.topk)).mean()
     
        label_mat2 = label_mat[tc_items - args.n_query].long().cpu().numpy()
        res_mat = pred_mat[tc_items - args.n_query].numpy() & label_mat2

        # tc_recall = (res_mat.sum(1) / label_mat.sum(1)).mean()
        tc_precision = (res_mat.sum(1) / max(args.topk)).mean()

    return {'ftp_precision@{}'.format(max(args.topk)) : ftp_precision, \
            'tc_precision@{}'.format(max(args.topk)) : tc_precision}





if __name__ == "__main__":
    # 1. 
    # data preparing
    args, g = build_taggnn_graph(args)
    label_mat, tc_label_mat, mask_mat, g, split_data = data_split_and_prepare(args, g)
    train_items, val_ftp_items, val_tc_items, test_ftp_items, test_tc_items = split_data
    all_train_items = train_items + val_tc_items + test_tc_items
    
    val_ftp_items = torch.tensor(val_ftp_items)
    val_tc_items = torch.tensor(val_tc_items)
    test_ftp_items = torch.tensor(test_ftp_items)
    test_tc_items = torch.tensor(test_tc_items)
    model = train_model(args, g, label_mat, tc_label_mat, mask_mat, all_train_items, val_ftp_items, val_tc_items)




    embed()
    input()



