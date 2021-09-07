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
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """


import logging
import os
import re
import sys
import torch
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from src.process.dataset import FinDataset, Split, get_labels, get_pos_labels, get_transitions
from torch.utils.data import Subset

from _transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, EvalPrediction, Trainer
from src.utils.args import get_args
from src.utils.files import make_dir, set_seeds, set_warnings, save_list, open_list
from src.utils.logger import get_logger, get_log_level, save_params, extend_res_summary, save_results_to_csv
from src.process.formatter import get_k_train_test_data, format_submission
from src.process.evaluater import align_predictions, is_exact_match

# Get args
model_args, data_args, training_args = get_args()

# Set items
set_seeds(training_args.seed)
set_warnings()

# Setup logging
log_file_name = datetime.now().strftime('logfile_%Y_%m_%d_%H_%M_%S.log')
log_save_path = f'{training_args.output_dir}/{model_args.model_name}/{training_args.seed}/{log_file_name}'.lower()
make_dir(log_save_path)
logger = get_logger(log_save_path, no_stdout=False, set_level=model_args.log_level)
logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    training_args.local_rank,
    training_args.device,
    training_args.n_gpu,
    bool(training_args.local_rank != -1),
    training_args.fp16,
)
# logger.info("Training/evaluation parameters: %s", training_args)


def initialise():
    labels = get_labels()
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    pos_labels = get_pos_labels('data/pos_tags.txt')
    pos_label_map: Dict[int, str] = {i: label for i, label in enumerate(pos_labels)}
    num_pos_labels = len(pos_labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        id2poslabel=pos_label_map,
        num_pos_labels=num_pos_labels,
        add_pos=model_args.add_pos,
        use_bilstm=model_args.use_bilstm,
        use_graph=model_args.use_graph,
        graph_purpose=model_args.graph_purpose,
        use_graphlstm=model_args.use_graphlstm,
        graph_hidden=model_args.graph_hidden,
        graph_out=model_args.graph_out,
        add_feats=model_args.add_feats,
        transitions_path=f'{data_args.data_dir}/train_transitions.txt',
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name,
        from_tf=bool(".ckpt" in model_args.model_name),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        FinDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            local_rank=training_args.local_rank,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        FinDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
            local_rank=training_args.local_rank,
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        FinDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            local_rank=training_args.local_rank
        )
        if training_args.do_predict
        else None
    )

    return train_dataset, eval_dataset, test_dataset, config, tokenizer, model, labels, label_map


def run_one():

    # Prepare inputs
    train_dataset, eval_dataset, test_dataset, config, tokenizer, model, labels, label_map = initialise()

    # Train, Val, Predict
    train(
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
        test_dataset=test_dataset, 
        config=config, 
        tokenizer=tokenizer, 
        model=model, 
        labels=labels, 
        label_map=label_map, 
        fold=None
        )


def run_kfolds():

    # Prepare inputs
    train_dataset, eval_dataset, test_dataset, config, tokenizer, model, labels, label_map = initialise()

    # Train, Val, Cross-Val, Predict
    indexes = list(range(0, len(train_dataset)))
    for fold in range(0, model_args.folds):
        train_ix, eval_ix = get_k_train_test_data(
            indexes, k=model_args.folds, seed=training_args.seed, fold=fold)
        cv_train_dataset = Subset(train_dataset, train_ix)
        cv_eval_dataset = Subset(train_dataset, eval_ix)
        txt_file_path = f'{training_args.output_dir}/{model_args.model_name}/{training_args.seed}/K{fold}_eval_ix.txt'
        save_list(eval_ix, txt_file_path)

        train(
            train_dataset=cv_train_dataset, 
            eval_dataset=eval_dataset, 
            test_dataset=test_dataset, 
            config=config, 
            tokenizer=tokenizer, 
            model=model, 
            labels=labels, 
            label_map=label_map, 
            fold=fold,
            eval_dataset2=cv_eval_dataset,
            )


def train(train_dataset, eval_dataset, test_dataset, config, tokenizer, model, \
    labels, label_map, fold=None, eval_dataset2=None):

    fold_folder = '' if fold is None else f'K{str(fold)}/'
    save_folder = f'{training_args.output_dir}/{model_args.model_name}/{training_args.seed}/{fold_folder}'

    # Load previous model
    if model_args.resume_training:
        # if we want to load past model
        if model_args.model_folder_path is None:
            # not specified, default to folder naming format
            model_folder_path = save_folder
        else:
            # if specified, we use the folder given
            model_folder_path = model_args.model_folder_path

        train_args_path = f'{model_folder_path}training_args.bin'
        model_path = f'{model_folder_path}pytorch_model.bin'

        if os.path.isfile(train_args_path):
            train_args = torch.load(train_args_path)
            logger.info(f'Loaded train params from "{train_args_path}": {train_args}')
        else:
            train_args = training_args
            logger.error(f'Unable to load train params from "{train_args_path}')

        if os.path.isfile(model_path):
            model_checkpoint = torch.load(model_path)
            model.load_state_dict(model_checkpoint)
            logger.info(f'Loaded model from "{model_path}"')
        else:
            logger.error(f'Unable to load model from "{model_path}"')
    else:
        train_args = training_args


    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list, preds_list_CE, out_label_list_CE = \
            align_predictions(p.predictions, p.label_ids, label_map)
        scores = {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
            "exact match": is_exact_match(out_label_list, preds_list)[1],
            "precision_CE": precision_score(out_label_list_CE, preds_list_CE),
            "recall_CE": recall_score(out_label_list_CE, preds_list_CE),
            "f1_CE": f1_score(out_label_list_CE, preds_list_CE),
            "exact match_CE": is_exact_match(out_label_list_CE, preds_list_CE)[1]
        }
        logger.info(f'Scores: {scores}')
        return scores


    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Training
    logger.info('-'*50)
    if train_dataset is not None:
        trainer.train(
            model_path=save_folder if os.path.isdir(save_folder) else None
        )
        trainer.save_model(save_folder)
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(save_folder)

        result = trainer.evaluate(train_dataset)
        # Save results
        extend_res_summary({f"{re.sub('/','_',fold_folder)}_Train_{k}":v for k,v in result.items()})
        output_eval_file = f'{save_folder}train_results.txt'
        with open(output_eval_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
        logger.info('-'*50)

    # Cross-Validation
    if eval_dataset2 is not None and train_args.local_rank in [-1, 0]:
        result = trainer.evaluate(eval_dataset2)
        # Save results
        extend_res_summary({f"{re.sub('/','_',fold_folder)}_CV_{k}":v for k,v in result.items()})
        output_eval_file = f'{save_folder}crossval_results.txt'
        with open(output_eval_file, "w") as writer:
            logger.info("***** Cross-validation results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
        logger.info('-'*50)
    
    # Evaluation
    if eval_dataset is not None and train_args.local_rank in [-1, 0]:
        predictions, label_ids, result = trainer.predict(eval_dataset)
        # Save results
        extend_res_summary({f"{re.sub('/','_',fold_folder)}_Val_{k}":v for k,v in result.items()})
        output_eval_file = f'{save_folder}val_results.txt'
        with open(output_eval_file, "w") as writer:
            logger.info("***** Validation results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
        
        # Save predictions
        preds_list, _, _, _ = align_predictions(predictions, label_ids, label_map)
        save_predictions(preds_list, save_folder, mode='val')
        logger.info('-'*50)

    # Predict
    if test_dataset is not None and train_args.local_rank in [-1, 0]:
        # Save predictions
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list, _, _, _  = align_predictions(predictions, label_ids, label_map)
        save_predictions(preds_list, save_folder, mode='test')
        # output_test_results_file = f'{save_folder}test_results.txt'
        # with open(output_test_results_file, "w") as writer:
        #     for key, value in metrics.items():
        #         logger.info("  %s = %s", key, value)
        #         writer.write("%s = %s\n" % (key, value))
        logger.info('-'*50)


def save_predictions(preds_list, save_folder, mode='test'):
    """
    Note: You cannot save cross-validation predictions because the number of examples 
    from ref and pred is different. You will need to combine train and cross-val and 
    sort by original ref order before feeding in as 'preds_list'.
    """
    
    ref_path = f'{data_args.data_dir}/{mode}'
    out_path = f'{save_folder}{mode}_predictions'
    
    with open(out_path+'.txt', "w", encoding='utf-8') as writer:
        with open(ref_path+'.txt', "r", encoding='utf-8') as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not preds_list[example_id]:
                        example_id += 1
                elif preds_list[example_id]:
                    output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                    writer.write(output_line)
                else:
                    logger.warning(
                        "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0]
                    )

    format_submission(
        pred_path=out_path+'.txt',
        ref_path=ref_path+'.csv',
        out_path=out_path+'.csv'
    )



def main():
    if model_args.folds is None:
        run_one()
    else:
        run_kfolds()


if __name__ == "__main__":
    """
    Example run in command line
    >>>>
    sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
    --data_dir data --labels data/labels.txt --model_name_or_path bert-base-cased \
    --output_dir outs --max_seq_length 350 --num_train_epochs 5 --per_gpu_train_batch_size 4 \
    --save_steps 3000 --seed 123 --do_train --do_eval --do_predict --overwrite_output_dir

    """
    logger.info('-- starting process')
    save_params((model_args, data_args, training_args), save_results=True)
    extend_res_summary({'logfile':log_save_path})
    main()
    save_results_to_csv(f'outs/results.csv')
    logger.info('-- complete process')
