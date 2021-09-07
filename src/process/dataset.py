# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import pickle
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from _transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.data.data import Data as GeomData
from torch_geometric.utils.convert import from_networkx
from collections import defaultdict
import stanza
en_nlp = stanza.Pipeline(
    lang='en',                                      # our dataset only has English
    tokenize_pretokenized=True,                     # recognises splits only by spaces
    processors='tokenize,mwt,pos,lemma,depparse'
)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """
    #### ADD POS ####
    guid: str
    words: List[str]
    pos_tags: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    pos_tag_ids: [List[int]]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    graph_data: GeomData = None


class Split(Enum):
    train = "train"
    dev = "val"
    test = "test"


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset
    from _transformers import torch_distributed_zero_first

    class FinDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            local_rank=-1,
        ):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            with torch_distributed_zero_first(local_rank):
                # Make sure only the first process in distributed training processes the dataset,
                # and the others will use the cache.

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logging.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logging.info(f"Creating features from dataset file at {data_dir}")
                    examples = read_examples_from_file(data_dir, mode)
                    # TODO clean up all this to leverage built-in features of tokenizers
                    self.features = convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        # xlnet has a cls token at the end
                        cls_token=tokenizer.cls_token if not bool(model_type in ["t5"]) else None,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token if not bool(model_type in ["t5"]) else "</s>",
                        sep_token_extra=bool(model_type in ["roberta"]),
                        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token_id=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                        mode=mode
                    )
                    if local_rank in [-1, 0]:
                        logging.info(f"Saving features into cached file {cached_features_file}")
                        torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    #### ADD POS ####
    file_path = os.path.join(data_dir, f"{mode}.txt")
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        pos_tags = []
        for line in f:
            if line == "\n":
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, pos_tags=pos_tags))
                    guid_index += 1
                    words = []
                    labels = []
                    pos_tags = []
            else:
                splits = line.split()
                # print(line[:-1].encode('utf-8'), len(splits), len(line.split()))
                assert len(splits) <= 3
                words.append(splits[0])
                pos_tags.append(splits[1])
                if len(splits) > 2:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("_")
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels, pos_tags=pos_tags))
    return examples


def create_one_hot(pos, max_pos=60):
    l = np.zeros(max_pos) #, dtype=int)
    l[pos] = 1
    return l


def doc_to_graph(doc, pos_tag_ids, label_ids=None, mode='train', key=0, num_pos_labels=52):
    """
    For each document/sentence, we can convert the dependency graph 
    from Stanza dependency parser into Networkx directed graph format.

    Labels are already converted into numbers!
    """
    pos_pads = 0 # padding tags
    G = nx.DiGraph() # directed graph
    word_id_to_pieces = defaultdict(list)

    counter = 0
    docs = [word for sent in doc.sentences for word in sent.words]

    for ix, pos in enumerate(pos_tag_ids):

        if int(pos)==pos_pads:
            features = create_one_hot(
                pos=int(pos_pads), 
                max_pos=num_pos_labels
            )
            add_dict = {
                'text': '##',   # follows on previous word
                'x': features,
                'ix': ix,
                'key': key
            }
        else:
            counter += 1
            word = docs[counter]
            features = create_one_hot(
                pos=int(pos), 
                max_pos=num_pos_labels
            )
            add_dict = {
                'text': word.text,
                'x': features,
                'ix': ix,
                'key': key
            }
        word_id_to_pieces[counter] += [ix] # append to list

        if mode != 'test' or label_ids is not None:
            add_dict['y'] = label_ids[ix]
        
        G.add_nodes_from([(ix, add_dict)])

    for word in docs:
        if word.head>0:
            for h in word_id_to_pieces[word.head]:
                for i in word_id_to_pieces[word.id]:
                    G.add_edge(h, i)
    
    return from_networkx(G)


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token_id=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    mode='test'
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    #### ADD POS ####
    pos_labels = get_pos_labels('data/pos_tags.txt')
    pos_map = {label: i for i, label in enumerate(pos_labels)}
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logging.info("Writing example %d of %d", ex_index, len(examples))

        #### ADD POS ####
        tokens = []
        label_ids = []
        pos_tag_ids = [] #pad_pos_tag = 0 (<pad>)
        pad_token_pos_id = 0

        for word, label, pos in zip(example.words, example.labels, example.pos_tags):
            word_tokens = tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                pos_tag_ids.extend([pos_map[pos]] + [pad_token_pos_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
            pos_tag_ids = pos_tag_ids[: (max_seq_length - special_tokens_count)]
            # seg_ids = seg_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        pos_tag_ids += [pad_token_pos_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            pos_tag_ids += [pad_token_pos_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token is not None:
            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                pos_tag_ids += [pad_token_pos_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                pos_tag_ids = [pad_token_pos_id] + pos_tag_ids
                segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token_id] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            pos_tag_ids = ([pad_token_pos_id] * padding_length) + pos_tag_ids
        else:
            input_ids += [pad_token_id] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            pos_tag_ids += [pad_token_pos_id] * padding_length

        if len(input_ids) != max_seq_length:
            print('error:', input_ids)
            print(padding_length)
            print(max_seq_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(pos_tag_ids) == max_seq_length

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        # get graph info
        graph_data = doc_to_graph(
            doc=en_nlp(example.words), 
            pos_tag_ids=pos_tag_ids,
            label_ids=label_ids,
            mode=mode,
            key=example.guid,
            num_pos_labels=len(pos_map)
            )

        # review
        if ex_index < 5:
            logging.info("*** Example ***")
            logging.info("guid: %s", example.guid)
            logging.info("original: %s", " ".join(example.words))
            logging.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logging.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            if segment_ids is not None:
                logging.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logging.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logging.info("pos_tag_ids: %s", " ".join([str(x) for x in pos_tag_ids]))
            logging.info(f"graph_data: {graph_data}")

        # save final infos
        features.append(
            InputFeatures(
                input_ids=input_ids, 
                attention_mask=input_mask, 
                token_type_ids=segment_ids, 
                label_ids=label_ids, 
                pos_tag_ids=pos_tag_ids,
                graph_data=graph_data
                )
            )

    return features


def get_transitions() -> np.matrix:
    mat = np.loadtxt('data/train_transitions.txt')
    print(f"loaded transitions: {mat}")
    return mat
    

def get_labels() -> List[str]:
    print("get_labels:", ["_", "B-C", "I-C", "B-E", "I-E"])
    return ["_", "B-C", "I-C", "B-E", "I-E"]


def get_pos_labels(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()
    if "<pad>" not in labels:
        labels = ["<pad>"] + labels
    print('pos_labels:', labels)
    print('Number of pos_labels:', len(labels))
    return labels

