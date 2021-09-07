import logging
import numpy as np
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map: Dict) -> Tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    out_label_list_CE = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    preds_list_CE = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out = label_map[label_ids[i][j]]
                pred = label_map[preds[i][j]]
                out_label_list[i].append(out)
                out_label_list_CE[i].append(out[-1])
                preds_list[i].append(pred)
                preds_list_CE[i].append(pred[-1])

    return preds_list, out_label_list, preds_list_CE, out_label_list_CE


def is_exact_match(golds: List[int], preds: List[int]) -> Tuple[int, float]:
    cnt = 0
    for i in range(len(golds)):
        if golds[i] == preds[i]:
            cnt += 1
    logging.info(f'Number of exact matches: {cnt} out of {len(golds)}')
    return cnt, cnt/len(golds)