import logging
import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import SAGEConv, GINConv

logger = logging.getLogger(__name__)


class GraphTokenClassifier(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.1):
        super(GraphTokenClassifier, self).__init__()
        
        self.conv1 = SAGEConv(
            in_channels=in_channels,        # num_features
            out_channels=hidden_channels,
            normalize=False
            )
        self.conv2 = SAGEConv(
            in_channels=hidden_channels, 
            out_channels=out_channels,      # num_labels
            normalize=False
            )
        # self.conv1 = GINConv(Sequential(
        #             Linear(in_channels, hidden_channels),
        #             ReLU(),
        #             Linear(hidden_channels, hidden_channels),
        #             ReLU(),
        #             BatchNorm1d(hidden_channels),
        #         ))
        # self.conv2 = GINConv(Sequential(
        #             Linear(hidden_channels, hidden_channels),
        #             ReLU(),
        #             Linear(hidden_channels, out_channels),
        #             ReLU(),
        #             BatchNorm1d(out_channels),
        #         ))
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, x, edge_index):
        # Batch(batch=[1400], edge_index=[2, 313], ix=[1400], key=[4], ptr=[5], text=[4], x=[1400, 60], y=[1400])
        # Loaded graph: Batch(batch=[509], edge_index=[2, 501], key=[8], ptr=[9], text=[8], x=[509, 52], y=[509])
        
        # x: torch.Size([237, 52])
        logger.debug(f'x: {x.shape} --> {x}')
        # edge_index: torch.Size([2, 232])
        logger.debug(f'edge_index: {edge_index.shape} --> {edge_index}')

        x = self.conv1(x, edge_index)
        x = self.dropout(x.relu())
        output = self.conv2(x, edge_index)

        return output