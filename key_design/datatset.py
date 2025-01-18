from tokenizer import *
import json
import os
from settings import *


def get_dataset(all_data, tokenizer_max_length=64):
    all_codes = []

    for fdir in all_data:
        graph = json.load(open(os.path.join(DATA_PATH, fdir, "graph1.json")))
        info = json.load(open(os.path.join(DATA_PATH, fdir, "info.json")))
        for node in graph:
            if not node["isDel"]:
                continue

            processed_text = tokenize_by_punctuation(node["code"])
            tokens = tokenize_text(processed_text)
            all_codes.append(" ".join(tokens))

    return all_codes
