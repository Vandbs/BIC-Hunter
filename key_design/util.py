from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData
from typing import Dict, List, Union
from datatset import *
import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.nn import HANConv
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from settings import *
from torch_geometric.data import HeteroData
import json
import random
import sys
from itertools import chain

from genPyG import *
from genPairs import *
from genBatch import *
from model import *
from eval import *
import numpy as np
import os


def get_all_data():
    with open("../miniGraphs.json") as f:
        miniGraphs = json.load(f)

    data1 = json.load(open("../dataset1.json"))
    data2 = json.load(open("../dataset2.json"))
    data3 = json.load(open("../dataset3.json"))

    return (miniGraphs, data1, data2, data3)


def init_model(device, metadata):
    criterion = torch.nn.BCELoss()
    hanModel = HAN(device, 768, 768 * 2, metadata=metadata, heads=2, dropout=0.1)
    hanModel = hanModel.to(device)
    rankNetModel = rankNet(768 * 2)
    rankNetModel = rankNetModel.to(device)

    optimizer = torch.optim.Adam(
        chain(hanModel.parameters(), rankNetModel.parameters()), lr=1e-5
    )

    return hanModel, rankNetModel, optimizer, criterion


def divide_lst(lst, n, k):
    cnt = 0
    all_list = []
    for i in range(0, len(lst), n):
        if cnt < k - 1:
            all_list.append(lst[i : i + n])
        else:
            all_list.append(lst[i:])
            break
        cnt = cnt + 1
    return all_list


def get_sub_minigraphs(fdirs, all_minigraphs):
    sub_minigraphs = {}
    for fdir in fdirs:
        sub_minigraphs[fdir] = all_minigraphs[fdir]
    return sub_minigraphs


# used for k cross fold validation
def divide_minigraphs(all_minigraphs, k):
    all_fdirs = []
    for fdir in all_minigraphs.keys():
        all_fdirs.append(fdir)
    random.shuffle(all_fdirs)

    all_sub_minigraphs = []
    all_sub_fdirs = []
    for sub_fdirs in divide_lst(all_fdirs, int(len(all_fdirs) / k), k):
        if len(sub_fdirs) == 0:
            continue
        all_sub_fdirs.append(sub_fdirs)
        all_sub_minigraphs.append(get_sub_minigraphs(sub_fdirs, all_minigraphs))

    return all_sub_minigraphs, all_sub_fdirs


def get_all_batchlist(mini_graphs, k, max_pair):
    all_batch_list = []
    pair_cnt = 0
    all_sub_minigraphs, all_sub_fdirs = divide_minigraphs(mini_graphs, k)

    for sub_minigraph in all_sub_minigraphs:
        all_pairs = get_all_pairs(sub_minigraph, max_pair)
        pair_cnt = pair_cnt + len(all_pairs)
        batch_list = combinePair(all_pairs, 100)
        all_batch_list.append(batch_list)

    return all_batch_list, all_sub_fdirs, pair_cnt


def train_batchlist(batches, hanModel, rankNetModel, optimizer, criterion, device):
    all_loss = []
    hanModel.train()
    rankNetModel.train()

    for batch in batches:
        pyg1 = batch.pyg1.clone().to(device)
        pyg2 = batch.pyg2.clone().to(device)

        del_index1 = batch.del_index1.to(device)
        del_index2 = batch.del_index2.to(device)

        probs = batch.probs.to(device)
        x = hanModel(pyg1, del_index1)
        y = hanModel(pyg2, del_index2)

        optimizer.zero_grad()
        preds = rankNetModel(x, y)
        loss = criterion(preds, probs)
        loss.backward()
        optimizer.step()

        all_loss.append(loss.cpu().detach().item())

    return sum(all_loss)


def validate_batchlist(batches, hanModel, rankNetModel, criterion, device):
    all_loss = []
    hanModel.eval()
    rankNetModel.eval()

    for batch in batches:
        with torch.no_grad():
            pyg1 = batch.pyg1.clone().to(device)
            pyg2 = batch.pyg2.clone().to(device)

            del_index1 = batch.del_index1.to(device)
            del_index2 = batch.del_index2.to(device)

            probs = batch.probs.to(device)
            x = hanModel(pyg1, del_index1)
            y = hanModel(pyg2, del_index2)

            preds = rankNetModel(x, y)
            loss = criterion(preds, probs)
            all_loss.append(loss.cpu().detach().item())

    return sum(all_loss)
