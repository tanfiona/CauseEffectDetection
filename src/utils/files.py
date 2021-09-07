import json
import os
import csv
import pickle
import pandas as pd
import numpy as np
import warnings
import argparse
import torch
import random
from pathlib import Path
from _transformers import set_seed


def open_pickle(pkl_file_path):
    return pickle.load(open(pkl_file_path,"rb"))


def open_json(json_file_path, data_format=list):
    if data_format==dict or data_format=='dict':
        with open(json_file_path) as json_file:
            data = json.load(json_file)
    elif data_format==list or data_format=='list':
        data = []
        for line in open(json_file_path, encoding='utf-8'):
            data.append(json.loads(line))
    elif data_format==pd.DataFrame or data_format=='pd.DataFrame':
        data = pd.read_json(json_file_path, orient="records", lines=True)
    else:
        raise NotImplementedError
    return data


def save_json(ddict, json_file_path):
    with open(json_file_path, 'w') as fp:
        json.dump(ddict, fp)



def save_list(data:list, txt_file_path:str):
    with open(txt_file_path, "w") as f:
        for s in data:
            f.write(str(s) +"\n")


def open_list(txt_file_path:str):
    items = []
    with open(txt_file_path, "r") as f:
        for line in f:
            items.append(int(line.strip()))
    return items


def make_dir(save_path):
    path = Path(save_path)
    if not os.path.isdir(path.parent):
        os.makedirs(path.parent, exist_ok=True)


def set_seeds(seed):
    # for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def set_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
