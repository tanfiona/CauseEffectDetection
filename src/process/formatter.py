import sys
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, KFold


def get_k_train_test_data(indexes, k, seed=1234, fold=0):
    if k<=2:
        train_ix, test_ix = train_test_split(indexes, test_size=0.2, random_state=seed)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for i, (train_ix, test_ix) in enumerate(kf.split(indexes)):
            if i == fold:
                break
        train_ix = [indexes[ix] for ix in train_ix]
        test_ix = [indexes[ix] for ix in test_ix]
    logging.info(f'KFold {fold+1}/{k} split -->   ALL: {len(indexes)}   TRAIN: {len(train_ix)}   VAL: {len(test_ix)}')
    return train_ix, test_ix


def read_file(filename):
    text, tags = [], []
    with open(filename, encoding='utf-8') as f:
        t, ta = [], []
        for line in f.readlines():
            if line=="\n" and t and ta:
                text.append(t)
                tags.append(ta)
                t, ta = [], []
            else:
                splits = line.split()
                assert len(splits) <= 3
                t.append(splits[0])
                ta.append(splits[-1])
    
    logging.info(f'{filename}: {len(text)}')
    return text, tags

def write_file(output, data):
    with open(output, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(' '.join(line))
            f.write('\n')

def post_process(pred):
    post_pred = []
    for line in pred:
        post = []
        flag = {'E':0, 'C':0, '_':0}
        for idx, e in enumerate(line):
            if idx == 0 or idx == 1:
                post.append(e)
            elif idx == len(line)-2 or idx == len(line)-1 :
                post.append(post[-1])
            else:
                cnt = {'E':0, 'C':0, '_':0}
                for i in range(5):
                    cnt[line[idx-2+i]] += 1
                cnt = sorted(cnt.items(), key = lambda x:x[1], reverse=True)
                if cnt[0][0] == e and cnt[0][1] != cnt[1][1]:
                    post.append(e)
                else:
                    if e != post[-1] and cnt[0][1] == cnt[1][1] and not flag[e]: #and e != '_'
                        post.append(e)
                    elif e != post[-1] and cnt[0][1] >= 3 and cnt[0][1] == post[-1]:
                        post.append(post[-1])
                    else:
                        post.append(post[-1])
            flag[post[-1]] += 1
        post_pred.append(post)
    #write_file('test.txt', post_pred)
    for idx, line in enumerate(post_pred):
        c_pos = get_longest(line, 'C')
        e_pos = get_longest(line, 'E')
        post = np.empty(len(line), dtype=str)
        if c_pos:
            post[c_pos[0]: c_pos[1]+1] = 'C'
        if e_pos:
            post[e_pos[0]: e_pos[1]+1] = 'E'
        post[post == ''] = '_'
        post_pred[idx] = list(post)
    write_file('test1.txt', post_pred)
    return post_pred

def post_process_bio(pred):
    post_pred = []
    #write_file('test.txt', pred)
    for line in pred:
        post = ['' for _ in range(len(line))]
        flag = {'E':0, 'C':0}
        for idx, e in enumerate(line):
            if e == 'B-E' and flag['E'] < 2:
                post[idx] = 'B-E'
                flag['E'] += 1
                if flag['C'] == 1:
                    flag['C'] += 1
            elif e == 'I-E' and flag['E'] == 1:
                post[idx] = e
            elif e == 'B-C' and flag['C'] < 2:
                post[idx] = e
                flag['C'] += 1
                if flag['E'] == 1:
                    flag['E'] += 1
            elif e == 'I-C' and flag['C'] == 1:
                post[idx] = e
            elif e == '_' and flag['E'] == 1:
                post[idx] = e
                flag['E'] += 1
            elif e == '_' and flag['C'] == 1:
                post[idx] = e
                flag['C'] += 1
            else:
                post[idx] = '_'
            
        post_pred.append(post)
    #write_file('test.txt', post_pred)
    return post_pred

def is_equal_len(pred, ref):
    assert len(pred) == len(ref)
    for i in range(len(ref)):
        if len(pred[i]) != len(ref[i]):
            logging.info(f'{i}, {len(pred[i])}, {len(ref[i])}')
        assert len(pred[i]) == len(ref[i])
    return True

def evaluate(y_pred, y_test, label_map):
    truths = np.array([label_map[tag] for row in y_test for tag in row])
    predictions = np.array([label_map[tag] for row in y_pred for tag in row])  
    logging.info(f'Accuracy: {np.sum(truths == predictions) / len(truths)}')
    # # Print out the classification report
    logging.info('************************ classification report ***************************', '\t')
    logging.info(
        classification_report(
            truths, predictions,
            target_names=targets)
            )

    # # Print out task2 metrics
    logging.info('************************ tasks metrics ***************************', '\t')
    F1metrics = precision_recall_fscore_support(truths, predictions, average='weighted')
    logging.info(f'F1score: {F1metrics[2]}')
    logging.info(f'Precision: {F1metrics[1]}')
    logging.info(f'Recall: {F1metrics[0]}')
    cnt = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]:
            cnt += 1
    logging.info(f'Exact matches: {cnt/len(y_pred)} = {cnt} / {len(y_pred)} sentences')


def get_longest(line, tag):
    longest, p = [], [-1, 0]
    for idx, e in enumerate(line):
        if e == tag and p[0] == -1:
            p[0] = idx
        elif e == tag and idx == len(line)-1 and p[0] != -1:
            p[1] = idx
            longest.append(p)
        elif e != tag and p[0] != -1:
            p[1] = idx - 1
            longest.append(p)
            p = [-1, 0]
    longest = sorted(longest, key = lambda x:x[1]-x[0]+1, reverse=True)
    if len(longest):
        return longest[0]
    else:
        return None

def submit(sub, preds, texts):
    org_index = sub.values[:, 0]
    org_texts = sub.values[:, 1]
    cause, effect = ['' for i in range(len(preds))], ['' for i in range(len(preds))]
    for i, (text, pred, org_text) in enumerate(zip(texts, preds, org_texts)):
        d = {}
        index = 0
        for idx, w in enumerate(text):
            if isinstance(org_index[i], str) and idx == 0 and len(org_index[i].split('.'))==3:
                org_text = w + org_text
            index = org_text.find(w, index)
            if not index == -1:
                d[idx] = (w, index)
                index += len(w)
        # logging.info(idx, pred)
        pred = [p[-1] for p in pred]
        c_pos = get_longest(pred, 'C')
        e_pos = get_longest(pred, 'E')
        if c_pos:
            cause[i] = org_text[d[c_pos[0]][1] : (d[c_pos[1]][1] + len(d[c_pos[1]][0]))]
        if e_pos:
            effect[i] = org_text[d[e_pos[0]][1] : (d[e_pos[1]][1] + len(d[e_pos[1]][0]) )]
    # logging.info(cause[-1])
    # logging.info(effect[-1])
    # logging.info(len(cause), len(effect))
    return cause, effect


def format_submission(pred_path, ref_path, out_path):
    text, preds = read_file(pred_path)
    preds = post_process_bio(preds)
    sub = pd.read_csv(ref_path, delimiter=';')
    cause, effect = submit(sub, preds, text)
    sub['Cause'] = cause
    sub['Effect'] = effect
    sub.to_csv(out_path, ';', index=0)


if __name__ == "__main__":
    # python NTUNLP/formatter.py NTUNLP/outs/test_predictions.txt data/test.csv NTUNLP/outs/test_pred.csv
    pred_path, ref_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    format_submission(pred_path, ref_path, out_path)
