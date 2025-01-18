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
from util import *


if __name__ == "__main__":
    device = ""
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.enabled = False
    all_mini_graphs, data1, data2, data3 = get_all_data()
    test_set = data1
    trainset = []
    trainset.append(data2)
    trainset.append(data3)

    all_train_codes = get_dataset(trainset)
    tagged_data = [
        TaggedDocument(words=codes.split(), tags=[str(i)])
        for i, codes in enumerate(all_train_codes)
    ]
    doc2vec = Doc2Vec(tagged_data, vector_size=768, window=2, min_count=1, workers=4)

    all_mini_graphs, trainset, testset = get_all_data()

    dataset = trainset
    max_pair = 100

    mini_graphs = get_sub_minigraphs(dataset, all_mini_graphs)

    all_batch_list, all_sub_fdirs, pair_cnt = get_all_batchlist(
        mini_graphs, 1, max_pair=max_pair, doc2vec=doc2vec
    )

    fdirs = []
    for fdir in trainset:
        fdirs.append(fdir)

    for fdir in testset:
        fdirs.append(fdir)

    all_true_cid_map = get_true_cid_map(fdirs)
    dir_to_minigraphs = get_dir_to_minigraphs(
        get_sub_minigraphs(fdirs, all_mini_graphs), doc2vec
    )

    hanModel, rankNetModel, optimizer, criterion = init_model(
        device, all_batch_list[0][0].pyg1.metadata(), doc2vec
    )

    best_f1_score = 0
    epochs = 20

    all_info = []
    for epoch in range(epochs):
        total_train_loss = 0
        total_valid_loss = 0
        total_tp1 = 0
        total_fp1 = 0

        total_tp2 = 0
        total_fp2 = 0

        total_tp3 = 0
        total_fp3 = 0

        total_t = 0

        trainset = all_sub_fdirs[0]

        for j, batch_list in enumerate(all_batch_list):
            total_train_loss = total_train_loss + train_batchlist(
                batch_list, hanModel, rankNetModel, optimizer, criterion, device
            )

        eval(
            trainset,
            dir_to_minigraphs,
            hanModel,
            rankNetModel,
            device
        )
        tp1, fp1, t = eval_top(
            trainset,
            dir_to_minigraphs,
            hanModel,
            rankNetModel,
            device,
            all_true_cid_map,
            1,
        )
        tp2, fp2, t = eval_top(
            trainset,
            dir_to_minigraphs,
            hanModel,
            rankNetModel,
            device,
            all_true_cid_map,
            2,
        )
        tp3, fp3, t = eval_top(
            trainset,
            dir_to_minigraphs,
            hanModel,
            rankNetModel,
            device,
            all_true_cid_map,
            3,
        )
        total_t = total_t + t
        total_tp1 = total_tp1 + tp1
        total_fp1 = total_fp1 + fp1
        total_tp2 = total_tp2 + tp2
        total_fp2 = total_fp2 + fp2
        total_tp3 = total_tp3 + tp3
        total_fp3 = total_fp3 + fp3

        cur_f1_score = (
            2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
        ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
        info = {}
        info["epoch"] = epoch
        info["pair_cnt"] = pair_cnt
        info["train_loss"] = total_train_loss
        info["train_f1_score"] = (
            2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
        ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
        info["train_top1_precision"] = total_tp1 / (total_tp1 + total_fp1)
        info["train_top1_recall"] = total_tp1 / total_t
        info["train_top2_precision"] = total_tp2 / (total_tp2 + total_fp2)
        info["train_top2_recall"] = total_tp2 / total_t
        info["train_top3_precision"] = total_tp3 / (total_tp3 + total_fp3)
        info["train_top3_recall"] = total_tp3 / total_t

        total_tp1 = 0
        total_fp1 = 0
        total_tp2 = 0
        total_fp2 = 0
        total_tp3 = 0
        total_fp3 = 0
        total_t = 0
        eval(testset, dir_to_minigraphs, hanModel, rankNetModel, device)
        tp1, fp1, t = eval_top(
            testset,
            dir_to_minigraphs,
            hanModel,
            rankNetModel,
            device,
            all_true_cid_map,
            1,
        )
        tp2, fp2, t = eval_top(
            testset,
            dir_to_minigraphs,
            hanModel,
            rankNetModel,
            device,
            all_true_cid_map,
            2,
        )
        tp3, fp3, t = eval_top(
            testset,
            dir_to_minigraphs,
            hanModel,
            rankNetModel,
            device,
            all_true_cid_map,
            3,
        )
        total_t = total_t + t
        total_tp1 = total_tp1 + tp1
        total_fp1 = total_fp1 + fp1
        total_tp2 = total_tp2 + tp2
        total_fp2 = total_fp2 + fp2
        total_tp3 = total_tp3 + tp3
        total_fp3 = total_fp3 + fp3

        cur_f1_score = (
            2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
        ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
        info["test_f1_score"] = (
            2 * (total_tp1 / (total_tp1 + total_fp1)) * (total_tp1 / total_t)
        ) / (total_tp1 / (total_tp1 + total_fp1) + total_tp1 / total_t)
        info["test_top1_precision"] = total_tp1 / (total_tp1 + total_fp1)
        info["test_top1_recall"] = total_tp1 / total_t
        info["test_top2_precision"] = total_tp2 / (total_tp2 + total_fp2)
        info["test_top2_recall"] = total_tp2 / total_t
        info["test_top3_precision"] = total_tp3 / (total_tp3 + total_fp3)
        info["test_top3_recall"] = total_tp3 / total_t

        all_info.append(info)
        with open(f"without_codebert.json", "w") as f:
            json.dump(all_info, f)
