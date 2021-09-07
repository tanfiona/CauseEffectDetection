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
"""PyTorch RoBERTa model. """


import logging
import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .configuration_roberta import RobertaConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu, ViterbiDecoder
from .modeling_utils import create_position_ids_from_input_ids
from .modeling_graph import GraphTokenClassifier


logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://cdn.huggingface.co/roberta-base-pytorch_model.bin",
    "roberta-large": "https://cdn.huggingface.co/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://cdn.huggingface.co/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://cdn.huggingface.co/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://cdn.huggingface.co/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://cdn.huggingface.co/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        return super().forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds
        )

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


ROBERTA_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

ROBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.RobertaTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
"""


@add_start_docstrings(
    "The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    ROBERTA_START_DOCSTRING,
)
class RobertaModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


@add_start_docstrings("""RoBERTa Model with a `language modeling` head on top. """, ROBERTA_START_DOCSTRING)
class RobertaForMaskedLM(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForMaskedLM
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


@add_start_docstrings(
    """RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
    on top of the pooled output) e.g. for GLUE tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Roberta Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForMultipleChoice(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor`` of shape ``(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification loss.
        classification_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            `num_choices` is the second dimension of the input tensors. (see `input_ids` above).

            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForMultipleChoice
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMultipleChoice.from_pretrained('roberta-base')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

        """
        num_choices = input_ids.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.roberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings(
    """Roberta Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForTokenClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        # main options
        self.num_labels = config.num_labels
        self.pos_num_labels = config.num_pos_labels # Count <pad>(0)
        self.add_pos = config.add_pos
        self.use_bilstm = config.use_bilstm

        # graph options
        self.use_graph = config.use_graph
        self.graph_purpose = config.graph_purpose
        self.use_graphlstm = config.use_graphlstm
        self.add_feats = config.add_feats

        # layers & dims
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        hidden_size = config.hidden_size
        if config.add_pos:
            hidden_size += self.pos_num_labels-1

        if config.use_graph:

            ### IN_CHANNELS ###
            if config.add_feats:
                if config.graph_purpose=='post':
                    in_channels=config.num_labels+hidden_size
                else:
                    in_channels=hidden_size
            else:
                if config.graph_purpose=='post':
                    # post-processing takes pred labels as inputs
                    in_channels=config.num_labels
                else:
                    # no node features, default to one vector
                    in_channels=1

            ### OUT_CHANNELS ###
            if 'embed' in config.graph_purpose or config.use_graphlstm:
                # use graph as embeddings
                out_channels=config.graph_out
            else:
                # use graph as alt classifier or post processor
                out_channels=self.num_labels
            
            ### GRAPH-RELATED LAYERS ###
            self.gnn = GraphTokenClassifier(
                in_channels=in_channels, 
                out_channels=out_channels, 
                hidden_channels=config.graph_hidden,
                dropout=config.hidden_dropout_prob
                )

            if config.graph_purpose=='classifier':
                self.weightavg = nn.Linear(config.num_labels*2, config.num_labels)
            
            if config.use_graphlstm:
                # Set up dimensions
                if 'classifier' in config.graph_purpose:
                    graphlstm_out = 64
                    self.g_emission = nn.Linear(graphlstm_out, config.num_labels)
                else:
                    graphlstm_out = out_channels
                
                self.reader = nn.LSTM(
                    out_channels,
                    graphlstm_out//2,
                    bidirectional=True,
                    dropout=config.hidden_dropout_prob
                )
            
            ### GRAPH OUT --> HIDDEN_CHANNELS ###
            if 'embed' in config.graph_purpose:
                if config.graph_purpose=='embed':
                    hidden_size += out_channels
                elif config.graph_purpose=='embed2':
                    hidden_size = out_channels
                else:
                    logging.error('No such graph embed method!')

        if config.use_bilstm:
            self.lstm = nn.LSTM(
                hidden_size,
                hidden_size//2,
                bidirectional=True,
                dropout=config.hidden_dropout_prob
            )

        if not self.use_graph or self.graph_purpose!='classifier2':
            self.emission = nn.Linear(hidden_size, config.num_labels)
        
        # print('num_labels:', config.num_labels)
        # print('num_pos_labels:', config.num_pos_labels)
        self.transition_ = nn.Parameter(torch.Tensor(config.num_labels, config.num_labels))
        self.transition_.data.zero_()
        # Five Labels - transition matrix is counted using train.csv
        # self.transition_.data = torch.Tensor([
        #     [-1.0041e-01, -3.1015e+00, -float("Inf"), -2.9876e+00, -float("Inf")],
        #     [-6.6689e+00, -float("Inf"), -1.2706e-03, -float("Inf"), -float("Inf")],
        #     [-3.7743e+00, -float("Inf"), -3.8819e-02, -4.1915e+00, -float("Inf")],
        #     [-6.2634e+00,  -float("Inf"), -float("Inf"), -float("Inf"), -1.9066e-03],
        #     [-4.1109e+00, -5.6472e+00, -float("Inf"), -float("Inf"), -2.0122e-02]])
        if config.transitions_path is not None:
            self.transition_.data = torch.Tensor(np.loadtxt(config.transitions_path))

        self.decoder = ViterbiDecoder(config.label2id)

        logger.info(f"ADD_POS: {config.add_pos}, USE_GRAPH: {config.use_graph}, GRAPH_PURPOSE: {config.graph_purpose}")
        logger.info(f"TRANSITIONS: {self.transition_.data}")

        self.init_weights()

    def log_sum_exp(self, tensor, dim):
        """
        Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.
        :param tensor: tensor
        :param dim: dimension to calculate log-sum-exp of
        :return: log-sum-exp
        """
        m, _ = torch.max(tensor, dim)
        m_expanded = m.unsqueeze(dim).expand_as(tensor)
        return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        pos_tag_ids=None,
        graph_data=None,
        mode='default',
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import RobertaTokenizer, RobertaForTokenClassification
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForTokenClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size, seq_len = input_ids.shape

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logger.debug(f'bert output.shape: {sequence_output.shape}')

        if self.add_pos:
            pos_one_hot = torch.zeros(pos_tag_ids.size(0), pos_tag_ids.size(1), self.pos_num_labels).to(device)
            pos_one_hot = pos_one_hot.scatter_(2, pos_tag_ids.unsqueeze(2), 1)[:, :, 1:] # (batch_size, seq_len, num_pos_labels)
            sequence_output = torch.cat([sequence_output, pos_one_hot], dim=2)

        if self.use_graph and self.graph_purpose!='post':

            if self.add_feats:
                node_features=sequence_output.view(batch_size*seq_len, -1)
            else:
                node_features=torch.ones(batch_size*seq_len,1).to(device)

            graph_data = graph_data.to(device)
            logger.debug(f'using graph data: {graph_data}')            
            graph_output = self.gnn(
                x=node_features,
                edge_index=graph_data.edge_index
            ).view(batch_size,seq_len,-1)
            logger.debug(f'graph_output.shape: {graph_output.shape}')

            if self.use_graphlstm:
                graph_output = graph_output.permute(1, 0, 2)
                graph_output, (ht, ct) = self.reader(graph_output)
                graph_output = graph_output.permute(1, 0, 2)

                if 'classifier' in self.graph_purpose:
                    graph_output = self.g_emission(graph_output)
                
            if self.graph_purpose=='embed':
                sequence_output = torch.cat([sequence_output, graph_output], dim=2)
            elif self.graph_purpose=='embed2':
                sequence_output = graph_output
        
        if self.use_bilstm:
            # bert gives (batch_size, seq_len, hidden_size)
            # bert output.shape: torch.Size([4, 350, 768])
            # lstm expects inputs of (seq_len, batch_size, features)
            sequence_output = sequence_output.permute(1, 0, 2)
            sequence_output, (ht, ct) = self.lstm(sequence_output)
            sequence_output = sequence_output.permute(1, 0, 2)
            logger.debug(f'bilstm output.shape: {sequence_output.shape}')

        if not self.use_graph or self.graph_purpose!='classifier2':
            logits = self.emission(sequence_output) # out: (batch_size, seq_len, num_labels)
            logger.debug(f'logits.shape: {logits.shape}') # logits.shape: torch.Size([4, 350, 5])

        # If graph is used as co-classifier
        if self.use_graph and 'classifier' in self.graph_purpose:
            if self.graph_purpose=='classifier':
                # adjustable weightage
                logits = torch.cat([logits, graph_output], dim=2)
                logits = self.weightavg(logits)
            elif self.graph_purpose=='classifier2':
                # only use graph predictions
                logits = graph_output
            elif self.graph_purpose=='classifier3':
                # equal weightage
                logits = (logits + graph_logits)/2
            else:
                logging.error('No such graph classifier method!')
            logger.debug(f'combined logits.shape: {logits.shape}')
        
        # If graph is used for post processing
        if self.use_graph and self.graph_purpose=='post':
            graph_data = graph_data.to(device)
            
            if self.add_feats:
                # original pos shape: e.g. x=[1400, 60]
                node_features = torch.cat([
                    logits.view(batch_size*seq_len, -1), 
                    sequence_output.view(batch_size*seq_len, -1),
                    ], dim=1)
            else:
                node_features = logits.view(batch_size*seq_len, -1)

            logits = self.gnn(
                x=node_features,
                edge_index=graph_data.edge_index
            ).view(batch_size,seq_len,self.num_labels)
            logger.debug(f'graph logits.shape: {logits.shape}')

        # For Viterbi Decode
        if mode == 'test':
            logits = logits.unsqueeze(2).expand(logits.size(0), logits.size(1), self.num_labels, self.num_labels) # (batch_size, seq_len, num_labels, num_labels)
            logits = logits + self.transition_.unsqueeze(0).unsqueeze(0)  # (batch_size, seq_len, num_labels, num_labels)
        
        #outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        batch_size = logits.size(0)
        word_pad_len = logits.size(1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                if mode == 'train':
                    # print('train')
                    active_loss = attention_mask.view(-1) == 1
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    active_logits = logits.view(-1, self.num_labels)
                    loss = loss_fct(active_logits, active_labels)
                # For Viterbi Decode
                elif mode == 'test':
                    #'''
                    # print('test')
                    active_loss = attention_mask == 1
                    lengths_ = [torch.nonzero(active_loss[i])[-1].item()+1 for i in range(active_loss.size(0))]
                    # Sort by lengths
                    lengths, indices = torch.sort(torch.Tensor(lengths_), descending=True)
                    scores = logits.gather(0, indices.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(batch_size, word_pad_len, self.num_labels, self.num_labels).type_as(labels))
                    labels = labels.gather(0, indices.unsqueeze(1).expand(batch_size, word_pad_len).type_as(labels))

                    # Gold scores
                    scores_at_targets = torch.zeros(batch_size, word_pad_len).to(device)
                    for i in range(batch_size):
                        for j in range(word_pad_len):
                            if labels[i, j] != CrossEntropyLoss().ignore_index:
                                scores_at_targets[i, j] = scores.view(batch_size, word_pad_len, -1)[i, j, labels[i, j]]
                    from torch.nn.utils.rnn import pack_padded_sequence
                    scores_at_targets = pack_padded_sequence(scores_at_targets, lengths, batch_first=True)[0]
                    gold_score = scores_at_targets.sum()
                    # print('gold:\n', gold_score)

                    # All paths' scores
                    scores_upto_t = torch.zeros(scores.size(0), self.num_labels).to(device)
                    for t in range(int(max(lengths))):
                        batch_size_t = sum([l > t for l in lengths])
                        if t == 0:
                            scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, 0, :]  # (batch_size, num_labels)
                        else:
                            scores_upto_t[:batch_size_t] = self.log_sum_exp(
                                scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                                dim=1)  # (batch_size, num_labels)
                    all_paths_scores = scores_upto_t[:, 0].sum()
                    # print('all_paths_scores:\n', all_paths_scores)

                    viterbi_loss = all_paths_scores - gold_score
                    viterbi_loss = viterbi_loss / batch_size

                    loss = viterbi_loss
                    logits = self.decoder.decode(logits, lengths_)
                    # print('transition matrix:\n', self.transition.data)
                    # print('Loss:', loss)
                    #'''
            else:
                print("No attention mask!")
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + (logits,) + outputs
        else:
            if attention_mask is not None:
                print('in else')
                active_loss = attention_mask == 1
                lengths_ = [torch.nonzero(active_loss[i])[-1].item()+1 for i in range(active_loss.size(0))]
                logits = self.decoder.decode(logits, lengths_)
                outputs = (logits, ) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """Roberta Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    ROBERTA_START_DOCSTRING,
)
class RobertaForQuestionAnswering(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(ROBERTA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-start scores (before SoftMax).
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length,)`):
            Span-end scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        # The checkpoint roberta-large is not fine-tuned for question answering. Please see the
        # examples/question-answering/run_squad.py example to see how to fine-tune a model to a question answering task.

        from transformers import RobertaTokenizer, RobertaForQuestionAnswering
        import torch

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForQuestionAnswering.from_pretrained('roberta-base')

        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_ids = tokenizer.encode(question, text)
        start_scores, end_scores = model(torch.tensor([input_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])

        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)