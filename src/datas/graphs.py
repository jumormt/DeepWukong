from dataclasses import dataclass
import networkx as nx
from os.path import exists
from typing import List
import torch
from torch_geometric.data import Data
from src.vocabulary import Vocabulary


@dataclass(frozen=True)
class XFGNode:
    ln: int


@dataclass
class XFGEdge:
    from_node: XFGNode
    to_node: XFGNode


@dataclass
class XFG:
    def __init__(self, path: str = None, xfg: nx.DiGraph = None):
        if xfg is not None:
            xfg_nx: nx.DiGraph = xfg
        elif path is not None:
            assert exists(path), f"xfg {path} not exists!"
            xfg_nx: nx.DiGraph = nx.read_gpickle(path)
        else:
            raise ValueError("invalid inputs!")
        self.__init_graph(xfg_nx)

    def __init_graph(self, xfg_nx: nx.DiGraph):
        self.__nodes, self.__edges, self.__tokens_list = [], [], []
        self.__node_to_idx = {}
        k_to_nodes = {}
        for idx, n in enumerate(xfg_nx):
            tokens = xfg_nx.nodes[n]["code_sym_token"]
            xfg_node = XFGNode(ln=n)
            self.__tokens_list.append(tokens)
            self.__nodes.append(xfg_node)
            k_to_nodes[n] = xfg_node
            self.__node_to_idx[xfg_node] = idx
        for n in xfg_nx:
            for k in xfg_nx[n]:
                if xfg_nx[n][k]["c/d"] == "c":
                    self.__edges.append(
                        XFGEdge(from_node=k_to_nodes[n],
                                to_node=k_to_nodes[k]))
                elif xfg_nx[n][k]["c/d"] == "d":
                    self.__edges.append(
                        XFGEdge(from_node=k_to_nodes[n],
                                to_node=k_to_nodes[k]))
        self.__label = xfg_nx.graph["label"]

    @property
    def nodes(self) -> List[XFGNode]:
        return self.__nodes

    @property
    def edges(self) -> List[XFGEdge]:
        return self.__edges

    @property
    def label(self) -> int:
        return self.__label

    def to_torch(self, vocab: Vocabulary, max_len: int) -> Data:
        """Convert this graph into torch-geometric graph

        Args:
            vocab:
            max_len: vector max_len for node content
        Returns:
            :torch_geometric.data.Data
        """
        node_tokens = []
        for idx, n in enumerate(self.nodes):
            node_tokens.append(self.__tokens_list[idx])
        # [n_node, max seq len]
        node_ids = torch.full((len(node_tokens), max_len),
                              vocab.get_pad_id(),
                              dtype=torch.long)
        for tokens_idx, tokens in enumerate(node_tokens):
            ids = vocab.convert_tokens_to_ids(tokens)
            less_len = min(max_len, len(ids))
            node_ids[tokens_idx, :less_len] = torch.tensor(ids[:less_len],
                                                           dtype=torch.long)
        edge_index = torch.tensor(list(
            zip(*[[self.__node_to_idx[e.from_node],
                   self.__node_to_idx[e.to_node]] for e in self.edges])),
            dtype=torch.long)

        # save token to `x` so Data can calculate properties like `num_nodes`
        return Data(x=node_ids, edge_index=edge_index)
