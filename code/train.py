import torch
import torch.nn as nn
import torch.optim as optim

import argparse
from utils import *
from operation import *

def train_and_eval(args, arch, data, index):
    des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_relations, labels = data

    #number of different relations in edge_relations
    num_relations = edge_relations.max().item() + 1
    record = []
    test_record = []

    model = ModelOp(arch, args.hiddim, num_relations,labels.max().item() + 1, args.fdrop, args.drop,edge_index,edge_relations)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    for epoch in range(args.epochs):
        _train_and_eval(model, data, labels, criterion, optimizer, index, record, test_record)
    record.sort()
    record.reverse()
    return sum(record[:args.evals])/args.evals, max(test_record)

def _train_and_eval(model, data, labels, criterion, optimizer, index, record, test_record):
    des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_relations, labels = data
    des_tensor = des_tensor.cuda()
    tweets_tensor = tweets_tensor.cuda()
    num_prop = num_prop.cuda()
    category_prop = category_prop.cuda()
    
    model.train()
    optimizer.zero_grad()
    logits = model(des_tensor, tweets_tensor, num_prop, category_prop)
    idx_train, idx_val, idx_test = index
    loss_train = criterion(logits[idx_train], labels[idx_train])
    acc_train = accuracy(logits[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    
    model.eval()
    logits = model(des_tensor, tweets_tensor, num_prop, category_prop)
    acc_val = accuracy(logits[idx_val], labels[idx_val])
    acc_test = accuracy(logits[idx_test], labels[idx_test])

    #logging.info('loss_tra/val acc_ %f %f %f %f', loss_train, loss_val, acc_train, acc_val)
    record.append(acc_val.item())
    test_record.append(acc_test.item())