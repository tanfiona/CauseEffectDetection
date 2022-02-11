# CauseEffectDetection
Our paper is titled ["NUS-IDS at FinCausal 2021: Dependency Tree in Graph Neural Network for Better Cause-Effect Span Detection"](https://aclanthology.org/2021.fnp-1.6/).

### Abstract
Automatic identification of cause-effect spans in financial documents is important for causality modelling and understanding reasons that lead to financial events. To exploit the observation that words are more connected to other words with the same cause-effect type in a dependency tree, we construct useful graph embeddings by incorporating dependency relation features through a graph neural network. Our model builds on a [baseline BERT token classifier with Viterbi decoding (Kao et al., 2020)](https://github.com/pxpxkao/FinCausal-2020), and outperforms this baseline in cross-validation and during the competition. [In the official run of FinCausal 2021](https://competitions.codalab.org/competitions/33102#results), we obtained Precision, Recall, and F1 scores of 95.56%, 95.56% and 95.57% that all ranked 1st place, and an Exact Match score of 86.05% which ranked 3rd place.

### Poster
![Poster](https://github.com/tanfiona/CauseEffectDetection/blob/main/FinCausal_SharedTask_FNP_2021_POSTER.png)


# Running the Code
### Setting Up
Please install dependencies indicated under `environment.yml`.<br>
Download the datasets from the organisers and place under the `data` folder.<br>
Run `preprocess.py` script to format the datasets into the respective `txt` files.

### Train, Val, Predict
Our main code is under `run.py`. The main models are included under the `_transformers` folder. For example, our Proposed GNN module can be viewed under [`_transformers/modeling_graph.py`](https://github.com/tanfiona/CauseEffectDetection/blob/main/_transformers/modeling_graph.py).<br>

To obtain the submission predictions for the competition corresponding to Proposed in our paper, you may run the shell script `run_submission.sh`.
To train and validate to recreate KFold experiments in our paper, you may run all the remaining shell scripts.

# Cite Us
```
@inproceedings{tan-ng-2021-nus,
    title = "{NUS}-{IDS} at {F}in{C}ausal 2021: Dependency Tree in Graph Neural Network for Better Cause-Effect Span Detection",
    author = "Tan, Fiona Anting  and
      Ng, See-Kiong",
    booktitle = "Proceedings of the 3rd Financial Narrative Processing Workshop",
    month = "15-16 " # sep,
    year = "2021",
    address = "Lancaster, United Kingdom",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.fnp-1.6",
    pages = "37--43",
}
```
