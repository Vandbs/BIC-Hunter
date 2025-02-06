# BIC-Hunter

## Project Description
This repository provides both the dataset and the code used in our paper, **BIC-Hunter**.

In this work, we introduce **BIC-Hunter**, a deep learning model designed to detect root cause deletion lines within bug-fixing commits, which are then used as input for the **SZZ algorithm**. The process begins with the application of **Confidence Learning (CL)** to denoise the dataset, ensuring higher quality data. Following this, a **Graph Convolutional Network (GCN)** model is constructed, which effectively captures the semantic relationships between each deletion line and other related lines, including both deletions and additions. To accurately identify the root cause of the bug, **BIC-Hunter** employs a **learning-to-rank** technique to rank all deletion lines within the commit, prioritizing those most likely to represent the root cause.

## Environments

1. OS: Linux

   GPU: NVIDIA A6000.

2. Language: Python (v3.10)

3. CUDA: 11.8

4. Torch: 2.2.2
   
5. other packages please refer to `requirements.txt`

## Train & Test
In order to achieve the verification of model performance by BIC-Hunte, run `train.py`.

In order to achieve the performance test of a single component, revise `cofident.py` and `model.py` and modify the interface.

## Data

data from : Tang et al https://ieeexplore.ieee.org/document/10298347 

### graph data

Each directory in trainData contains three files. The file `info.json` contains the repository's name, the bug-fixing commit and the bug-inducing commit. The file `graph1.json` contains the heterogeneous graph generated for the bug-fixing commit in json format. The file `graph2.json` contain our manual annotation result.

### final dataset

The files `dataset1.json`,`dataset2.json` and `dataset3.json` contain the filtered datasets for the DATASET1, DATASET2 and DATASET3 in our paper respectively.
