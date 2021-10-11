from typing import List, cast
from os.path import join
from argparse import ArgumentParser
import os
from src.utils import unique_xfg_sym, split_list
import networkx as nx
from src.preprocess.symbolizer import clean_gadget
from tqdm import tqdm
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf, DictConfig
from multiprocessing import cpu_count, Manager, Pool, Queue
import functools
import dataclasses
from src.preprocess.symbolizer import tokenize_code_line

USE_CPU = cpu_count()


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/dwk.yaml",
                            type=str)
    return arg_parser


def unique_data(cweid: str, root: str):
    """
    unique raw data without symbolization
    Args:
        cweid:
        root:

    Returns:

    """
    XFG_root_path = join(root, cweid, "XFG")
    testcaseids = os.listdir(XFG_root_path)
    xfg_paths = list()
    for testcase in testcaseids:
        testcase_root_path = join(XFG_root_path, testcase)
        for k in ["arith", "array", "call", "ptr"]:
            k_root_path = join(testcase_root_path, k)
            xfg_ps = os.listdir(k_root_path)
            for xfg_p in xfg_ps:
                xfg_path = join(k_root_path, xfg_p)
                xfg_paths.append(xfg_path)
    # remove duplicates and conflicts
    xfg_dict = unique_xfg_sym(xfg_paths)
    xfg_unique_paths = list()
    for md5 in xfg_dict:
        # remove conflicts
        if xfg_dict[md5]["label"] != -1:
            xfg_unique_paths.append(xfg_dict[md5]["xfg"])
    return xfg_unique_paths


@dataclasses.dataclass
class QueueMessage:
    XFG: nx.DiGraph
    xfg_path: str
    to_remove: bool = False
    is_finished: bool = False


def handle_queue_message(queue: Queue):
    """

    Args:
        queue:

    Returns:

    """
    xfg_ct = 0
    while True:
        message: QueueMessage = queue.get()
        if message.is_finished:
            break
        if message.to_remove:
            os.system(f"rm {message.xfg_path}")
        else:
            if message.XFG is not None:
                nx.write_gpickle(message.XFG, message.xfg_path)
                xfg_ct += 1
    return xfg_ct


def process_parallel(testcaseid: str, queue: Queue, XFG_root_path: str, split_token: bool):
    """

    Args:
        testcaseid:
        queue:
        XFG_root_path:

    Returns:

    """
    testcase_root_path = join(XFG_root_path, testcaseid)
    for k in ["arith", "array", "call", "ptr"]:
        k_root_path = join(testcase_root_path, k)
        xfg_ps = os.listdir(k_root_path)
        for xfg_p in xfg_ps:
            xfg_path = join(k_root_path, xfg_p)
            xfg: nx.DiGraph = nx.read_gpickle(xfg_path)
            for idx, n in enumerate(xfg):
                if "code_sym_token" in xfg.nodes[n]:
                    return testcaseid
            file_path = xfg.graph["file_paths"][0]
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_contents = f.readlines()
            code_lines = list()
            for n in xfg:
                code_lines.append(file_contents[n - 1])
            sym_code_lines = clean_gadget(code_lines)

            to_remove = list()
            for idx, n in enumerate(xfg):
                xfg.nodes[n]["code_sym_token"] = tokenize_code_line(sym_code_lines[idx], split_token)
                if len(xfg.nodes[n]["code_sym_token"]) == 0:
                    to_remove.append(n)
            xfg.remove_nodes_from(to_remove)

            if len(xfg.nodes) != 0:
                queue.put(QueueMessage(xfg, xfg_path))
            else:
                queue.put(QueueMessage(xfg, xfg_path, to_remove=True))
    return testcaseid


def add_symlines(cweid: str, root: str, split_token: bool):
    """

    Args:
        cweid:
        root:

    Returns:

    """

    XFG_root_path = join(root, cweid, "XFG")
    testcaseids = os.listdir(XFG_root_path)
    testcase_len = len(testcaseids)

    with Manager() as m:
        message_queue = m.Queue()  # type: ignore
        pool = Pool(USE_CPU)
        xfg_ct = pool.apply_async(handle_queue_message, (message_queue,))
        process_func = functools.partial(process_parallel, queue=message_queue, XFG_root_path=XFG_root_path,
                                         split_token=split_token)
        testcaseids_done: List = [
            testcaseid
            for testcaseid in tqdm(
                pool.imap_unordered(process_func, testcaseids),
                desc=f"testcases: ",
                total=testcase_len,
            )
        ]

        message_queue.put(QueueMessage(None, None, False, True))
        pool.close()
        pool.join()
    print(f"total {xfg_ct.get()} XFGs!")


if __name__ == '__main__':
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    config = cast(DictConfig, OmegaConf.load(__args.config))
    seed_everything(config.seed, workers=True)
    add_symlines(config.dataset.name, config.data_folder, config.split_token)
    xfg_unique_paths = unique_data(config.dataset.name, config.data_folder)
    split_list(xfg_unique_paths, join(config.data_folder, config.dataset.name))
