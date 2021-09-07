
"""
Get arguments

# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from _transformers import HfArgumentParser, TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name: str = field(
        default=None, metadata={"help": "Model identifier from huggingface.co/models"}
    )
    model_folder_path: str = field(
        default=None, metadata={"help": "Path to pretrained model folder "+\
        "(Should hold: pytorch_model.bin, training_args.bin). If not specified, "+\
            "we default to folder with 'out_folder/model_name/' naming."}
    )
    resume_training: bool = field(
        default=False, metadata={"help": "Set this flag to continue training from "+\
            "available model in directory (See model_folder_path)"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(
        default=False, metadata={"help": "Set this flag to use fast tokenization."}
    )
    folds: int = field(
        default=None, metadata={"help": "Number of folds to run on train set. If not specified, no KFolds is performed."}
    )
    add_pos: bool = field(
        default=False, metadata={"help": "Whether to use POS tag features."}
    )
    use_bilstm: bool = field(
        default=False, metadata={"help": "Set this flag to use BiLSTM in model."}
    )
    use_graph: bool = field(
        default=False, metadata={"help": "Set this flag to use GNN/GraphNetwork in model."}
    )
    graph_purpose: Optional[str] = field(
        default='classifier', metadata={"help": "Usage of GNN in model; Options: classifier, embed, embed2, post"}
    )
    use_graphlstm: bool = field(
        default=False, metadata={"help": "Set this flag to use Graph as LSTM Embeddings in model. Cannot use with graph_purpose='post'."}
    )
    add_feats: bool = field(
        default=False, metadata={"help": "Whether to use BERT(+POS) features in Graph nodes."}
    )
    graph_hidden: int = field(
        default=1024, metadata={"help": "Hidden dimension in GNN."}
    )
    graph_out: int = field(
        default=512, metadata={"help": "Output dimension in GNN. Only applicable if graph_purpose!='classifier' where it defaults to num_labels."}
    )
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    log_level: Optional[str] = field(
        default='info', metadata={"help": "Log level"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        default='data', metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    train_filename: Optional[str] = field(
        default=None, metadata={"help": "Train file name if not just 'train'"}
    )
    test_filename: Optional[str] = field(
        default=None, metadata={"help": "Test file name if not just 'test'"}
    )
    val_filename: Optional[str] = field(
        default=None, metadata={"help": "Val file name if not just 'val'"}
    )
    labels: Optional[str] = field(
        default=None, metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."}
    )
    max_seq_length: int = field(
        default=350,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_args():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    return model_args, data_args, training_args