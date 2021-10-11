import hashlib
from warnings import filterwarnings
import subprocess

from sklearn.model_selection import train_test_split
from typing import List, Union, Dict, Tuple
import numpy
import torch
import json
import os
import networkx as nx
from os.path import exists
from tqdm import tqdm

PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
BOS = "<BOS>"
EOS = "<EOS>"


def getMD5(s):
    '''
    得到字符串s的md5加密后的值

    :param s:
    :return:
    '''
    hl = hashlib.md5()
    hl.update(s.encode("utf-8"))
    return hl.hexdigest()


def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.trainer.data_loading",
                   lineno=102)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.utilities.data",
                   lineno=41)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=216)  # save
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=234)  # load
    filterwarnings("ignore",
                   category=DeprecationWarning,
                   module="pytorch_lightning.metrics.__init__",
                   lineno=43)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch._tensor",
                   lineno=575)
    filterwarnings("ignore",
                   category=UserWarning,
                   module="src.models.modules.common_layers",
                   lineno=0)


def count_lines_in_file(file_path: str) -> int:
    command_result = subprocess.run(["wc", "-l", file_path],
                                    capture_output=True,
                                    encoding="utf-8")
    if command_result.returncode != 0:
        raise RuntimeError(
            f"Counting lines in {file_path} failed with error\n{command_result.stderr}"
        )
    return int(command_result.stdout.split()[0])



def unique_xfg_raw(xfg_path_list):
    """f
    unique xfg from xfg list
    Args:
        xfg_path_list:

    Returns:
        md5_dict: {xfg md5:{"xfg": xfg_path, "label": 0/1/-1}}, -1 stands for conflict
    """
    md5_dict = dict()
    mul_ct = 0
    conflict_ct = 0

    for xfg_path in xfg_path_list:
        xfg = nx.read_gpickle(xfg_path)
        label = xfg.graph["label"]
        file_path = xfg.graph["file_paths"][0]
        assert exists(file_path), f"{file_path} not exists!"
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            file_contents = f.readlines()
        for ln in xfg:
            ln_md5 = getMD5(file_contents[ln - 1])
            xfg.nodes[ln]["md5"] = ln_md5
        edges_md5 = list()
        for edge in xfg.edges:
            edges_md5.append(xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"])
        xfg_md5 = getMD5(str(sorted(edges_md5)))
        if xfg_md5 not in md5_dict:
            md5_dict[xfg_md5] = dict()
            md5_dict[xfg_md5]["label"] = label
            md5_dict[xfg_md5]["xfg"] = xfg_path
        else:
            md5_label = md5_dict[xfg_md5]["label"]
            if md5_label != -1 and md5_label != label:
                conflict_ct += 1
                md5_dict[xfg_md5]["label"] = -1
            else:
                mul_ct += 1
    print(f"total conflit: {conflict_ct}")
    print(f"total multiple: {mul_ct}")
    return md5_dict


def unique_xfg_sym(xfg_path_list):
    """f
    unique xfg from xfg list
    Args:
        xfg_path_list:

    Returns:
        md5_dict: {xfg md5:{"xfg": xfg_path, "label": 0/1/-1}}, -1 stands for conflict
    """
    md5_dict = dict()
    mul_ct = 0
    conflict_ct = 0

    for xfg_path in tqdm(xfg_path_list, total=len(xfg_path_list), desc="xfgs: "):
        xfg = nx.read_gpickle(xfg_path)
        label = xfg.graph["label"]
        file_path = xfg.graph["file_paths"][0]
        assert exists(file_path), f"{file_path} not exists!"
        for ln in xfg:
            ln_md5 = getMD5(str(xfg.nodes[ln]["code_sym_token"]))
            xfg.nodes[ln]["md5"] = ln_md5
        edges_md5 = list()
        for edge in xfg.edges:
            edges_md5.append(xfg.nodes[edge[0]]["md5"] + "_" + xfg.nodes[edge[1]]["md5"])
        xfg_md5 = getMD5(str(sorted(edges_md5)))
        if xfg_md5 not in md5_dict:
            md5_dict[xfg_md5] = dict()
            md5_dict[xfg_md5]["label"] = label
            md5_dict[xfg_md5]["xfg"] = xfg_path
        else:
            md5_label = md5_dict[xfg_md5]["label"]
            if md5_label != -1 and md5_label != label:
                conflict_ct += 1
                md5_dict[xfg_md5]["label"] = -1
            else:
                mul_ct += 1
    print(f"total conflit: {conflict_ct}")
    print(f"total multiple: {mul_ct}")
    return md5_dict


def split_list(files: List[str], out_root_path: str):
    """

    Args:
        files:
        out_root_path:

    Returns:

    """
    X_train, X_test = train_test_split(files, test_size=0.2)
    X_test, X_val = train_test_split(X_test, test_size=0.5)
    if not exists(f"{out_root_path}"):
        os.makedirs(f"{out_root_path}")
    with open(f"{out_root_path}/train.json", "w") as f:
        json.dump(X_train, f)
    with open(f"{out_root_path}/test.json", "w") as f:
        json.dump(X_test, f)
    with open(f"{out_root_path}/val.json", "w") as f:
        json.dump(X_val, f)
