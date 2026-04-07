from argparse import ArgumentParser

import torch

from src.data_generator import build_PDG, build_XFG
from src.models.vd import DeepWuKong
from src.datas.graphs import XFG
import networkx as nx
from src.preprocess.symbolizer import clean_gadget, tokenize_code_line
from src.utils import filter_warnings
from torch_geometric.data import Batch


def add_syms(xfg: nx.DiGraph, split_token: bool):
    """

    Args:
        xfg:
        split_token:

    Returns:

    """
    file_path = xfg.graph["file_paths"][0]
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        file_contents = f.readlines()
    code_lines = list()
    for n in xfg:
        code_lines.append(file_contents[n - 1])
    sym_code_lines = clean_gadget(code_lines)
    for idx, n in enumerate(xfg):
        xfg.nodes[n]["code_sym_token"] = tokenize_code_line(sym_code_lines[idx], split_token)
    return xfg


if __name__ == '__main__':
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-c",
                              "--check-point",
                              help="checkpoint path",
                              type=str)
    __arg_parser.add_argument("-t",
                              "--target",
                              help="code csv root path",
                              type=str)
    __arg_parser.add_argument("-s",
                              "--source",
                              help="source code path",
                              type=str)
    __args = __arg_parser.parse_args()
    filter_warnings()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    model = DeepWuKong.load_from_checkpoint(checkpoint_path=__args.check_point).to(device)
    # load config and vocab
    config = model.hparams["config"]
    vocab = model.hparams["vocab"]
    vocab_size = vocab.get_vocab_size()
    pad_idx = vocab.get_pad_id()
    # preprocess for the code to detect
    PDG, key_line_map = build_PDG(__args.target, f"{config.data_folder}/sensiAPI.txt",
                                  __args.source)
    xfg_dict = build_XFG(PDG, key_line_map)
    Datas, meta_datas = list(), list()
    idx_to_xfg = dict()
    ct = 0
    for k in xfg_dict:
        for xfg in xfg_dict[k]:
            xfg_sym = add_syms(xfg, config.split_token)
            Datas.append(XFG(xfg=xfg_sym).to_torch(vocab, config.dataset.token.max_parts))
            meta_datas.append((xfg_sym.graph["file_paths"][0], xfg_sym.graph["key_line"]))
            idx_to_xfg[ct] = xfg_sym
            ct += 1
    # predict
    batch = Batch.from_data_list(Datas).to(device)
    logits = model(batch)
    _, preds = logits.max(dim=1)
    batched_res = zip(meta_datas, preds.tolist())
    for res in batched_res:
        if res[1] == 1:
            print(res[0])
