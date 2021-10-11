from omegaconf import DictConfig
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, GCNConv, GINEConv, GATv2Conv, GatedGraphConv, GlobalAttention
import torch.nn.functional as F
from src.vocabulary import Vocabulary
from src.models.modules.common_layers import STEncoder


class GraphConvEncoder(torch.nn.Module):
    """

    Kipf and Welling: Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    (https://arxiv.org/pdf/1609.02907.pdf)

    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GraphConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = GCNConv(config.rnn.hidden_size, config.hidden_size)

        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)

        for i in range(config.n_hidden_layers - 1):
            setattr(self, f"hidden_GCL{i}",
                    GCNConv(config.hidden_size, config.hidden_size))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(config.hidden_size,
                            ratio=config.pooling_ratio))

        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))

    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden]
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        node_embedding, edge_index, _, batch, _, _ = self.input_GPL(node_embedding, edge_index, None,
                                                                    batch)
        # [n_XFG; XFG hidden dim]
        out = self.attpool(node_embedding, batch)
        for i in range(self.__config.n_hidden_layers - 1):
            node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index))
            node_embedding, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, None, batch)
            out += self.attpool(node_embedding, batch)
        # [n_XFG; XFG hidden dim]
        return out


class GatedGraphConvEncoder(torch.nn.Module):
    """

    from Li et al.: Gated Graph Sequence Neural Networks (ICLR 2016)
    (https://arxiv.org/pdf/1511.05493.pdf)

    """

    def __init__(self, config: DictConfig, vocab: Vocabulary,
                 vocabulary_size: int,
                 pad_idx: int):
        super(GatedGraphConvEncoder, self).__init__()
        self.__config = config
        self.__pad_idx = pad_idx
        self.__st_embedding = STEncoder(config, vocab, vocabulary_size, pad_idx)

        self.input_GCL = GatedGraphConv(out_channels=config.hidden_size, num_layers=config.n_gru)

        self.input_GPL = TopKPooling(config.hidden_size,
                                     ratio=config.pooling_ratio)

        for i in range(config.n_hidden_layers - 1):
            setattr(self, f"hidden_GCL{i}",
                    GatedGraphConv(out_channels=config.hidden_size, num_layers=config.n_gru))
            setattr(
                self, f"hidden_GPL{i}",
                TopKPooling(config.hidden_size,
                            ratio=config.pooling_ratio))
        self.attpool = GlobalAttention(torch.nn.Linear(config.hidden_size, 1))

    def forward(self, batched_graph: Batch):
        # [n nodes; rnn hidden]
        node_embedding = self.__st_embedding(batched_graph.x)
        edge_index = batched_graph.edge_index
        batch = batched_graph.batch
        node_embedding = F.relu(self.input_GCL(node_embedding, edge_index))
        node_embedding, edge_index, _, batch, _, _ = self.input_GPL(node_embedding, edge_index, None,
                                                                    batch)
        # [n_XFG; XFG hidden dim]
        out = self.attpool(node_embedding, batch)
        for i in range(self.__config.n_hidden_layers - 1):
            node_embedding = F.relu(getattr(self, f"hidden_GCL{i}")(node_embedding, edge_index))
            node_embedding, edge_index, _, batch, _, _ = getattr(self, f"hidden_GPL{i}")(
                node_embedding, edge_index, None, batch)
            out += self.attpool(node_embedding, batch)
        # [n_XFG; XFG hidden dim]
        return out
