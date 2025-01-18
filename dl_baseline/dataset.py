from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import GloVe
from tokenizer import tokenize_by_punctuation
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import BertModel, BertTokenizer

class dl_dataset(Dataset):
    def __init__(self, all_nodes, all_codes, all_labels, all_fdirs, all_info):
        self.all_nodes = all_nodes
        self.all_codes = all_codes
        self.all_labels = all_labels
        self.all_fdirs = all_fdirs
        self.all_info = all_info
        self.glove = GloVe(name="6B", dim=300, cache="./glove/")
        self.tokenizer = BertTokenizer.from_pretrained('microsoft/codebert-base')
        self.model = BertModel.from_pretrained('microsoft/codebert-base')
    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, index):
        tokens = tokenize_by_punctuation(self.all_codes[index].lower()).split(" ")
        x = self.glove.get_vecs_by_tokens(tokens)
        label = self.all_labels[index]

        return x, label
