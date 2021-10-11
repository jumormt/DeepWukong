from dataclasses import dataclass
from typing import List
from torch_geometric.data import Data, Batch
import torch


@dataclass
class XFGSample:
    graph: Data
    label: int


class XFGBatch:
    def __init__(self, XFGs: List[XFGSample]):

        self.labels = torch.tensor([XFG.label for XFG in XFGs],
                                   dtype=torch.long)
        self.graphs = []
        for XFG in XFGs:
            self.graphs.append(XFG.graph)
        self.graphs = Batch.from_data_list(self.graphs)
        self.sz = len(XFGs)

    def __len__(self):
        return self.sz

    def pin_memory(self) -> "XFGBatch":
        self.labels = self.labels.pin_memory()
        self.graphs = self.graphs.pin_memory()
        return self

    def move_to_device(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.graphs = self.graphs.to(device)