from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.nn import HANConv

from torch_geometric.data import HeteroData
import json
import random

from genPyG import get_graph_data


class miniGraph:
    def __init__(self, g, pyg, fDir):
        self.g = g
        self.pyg = pyg
        self.fDir = fDir
        self.score = 0.0


# get all train pairs
def get_all_pairs(all_data_map, mask, max_cnt=1000):    #制作每生成图对，并赋予ground truth probability
    all_pairs = []
    mask_cnt1 = 0
    mask_cnt2 = 0
    for key, graphs in all_data_map.items():
        cnt = 0
        graphs1 = []
        for graph in graphs:  #graph是一个代码行的生成图
            if graph[0]["rootcause"] and mask[mask_cnt1]:
                graphs1.append(graph)
            mask_cnt1 = mask_cnt1 + 1

        for graph in graphs:
            if not graph[0]["rootcause"] and mask[mask_cnt2]:
                graphs1.append(graph)
            mask_cnt2 = mask_cnt2 + 1

        #先遍历一个修复提交下的所有代码行，将代码行的生成图的初始节点是引入bug的放前面排序

        for i in range(len(graphs1)):    #双重遍历一个修复提交下的所有代码行的生成图
            for j in range(i + 1, len(graphs1)):
                pyg1 = get_graph_data(graphs1[i])   #获取该代码行的生成图的所有信息
                pyg2 = get_graph_data(graphs1[j])
                minig1 = miniGraph(graphs1[i], pyg1, key)    #统一做成类
                minig2 = miniGraph(graphs1[j], pyg2, key)
                if graphs1[i][0]["rootcause"] == graphs1[j][0]["rootcause"]:   #制作每个生成图对的ground truth probability，用于后续计算损失
                    all_pairs.append({"x": minig1, "y": minig2, "prob": 0.5})
                    cnt = cnt + 1
                if graphs1[i][0]["rootcause"] and not graphs1[j][0]["rootcause"]:
                    all_pairs.append({"x": minig1, "y": minig2, "prob": 1.0})
                    cnt = cnt + 1
                if not graphs1[i][0]["rootcause"] and graphs1[j][0]["rootcause"]:
                    all_pairs.append({"x": minig1, "y": minig2, "prob": 0.0})
                    cnt = cnt + 1
                if cnt > max_cnt:
                    break
            if cnt > max_cnt:
                break

    # print("mask", mask.shape)
    # print("mask_cnt", mask_cnt1)

    return all_pairs


def get_dir_to_minigraphs(all_data_map):   #第i次修复提交存储所有代码行的图
    dir_to_minigraphs = {}
    for key, graphs in all_data_map.items():
        dir_to_minigraphs[key] = []
        for i in range(len(graphs)):   #遍历所有代码行的图
            pyg = get_graph_data(graphs[i])
            minig = miniGraph(graphs[i], pyg, key)
            dir_to_minigraphs[key].append(minig)
    return dir_to_minigraphs
