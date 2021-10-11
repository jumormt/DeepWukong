from torch.utils.data import Dataset
from omegaconf import DictConfig
from src.datas.graphs import XFG
from src.datas.samples import XFGSample
from os.path import exists
import json
from src.vocabulary import Vocabulary


class XFGDataset(Dataset):
    def __init__(self, XFG_paths_json: str, config: DictConfig, vocab: Vocabulary) -> None:
        """
        Args:
            XFG_root_path: json file of list of XFG paths
        """
        super().__init__()
        self.__config = config
        assert exists(XFG_paths_json), f"{XFG_paths_json} not exists!"
        with open(XFG_paths_json, "r") as f:
            __XFG_paths_all = list(json.load(f))
        self.__vocab = vocab
        self.__XFGs = list()
        for xfg_path in __XFG_paths_all:
            xfg = XFG(path=xfg_path)
            # if len(xfg.nodes) != 0:
            self.__XFGs.append(xfg)
        self.__n_samples = len(self.__XFGs)

    def __len__(self) -> int:
        return self.__n_samples

    def __getitem__(self, index) -> XFGSample:
        xfg: XFG = self.__XFGs[index]
        return XFGSample(graph=xfg.to_torch(self.__vocab,
                                            self.__config.dataset.token.max_parts),
                         label=xfg.label)

    def get_n_samples(self):
        return self.__n_samples
