from argparse import ArgumentParser
from typing import cast

from commode_utils.common import print_config
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from src.datas.datamodules import XFGDataModule
from src.models.vd import DeepWuKong
from src.train import train
from src.utils import filter_warnings, PAD
from src.vocabulary import Vocabulary


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c",
                            "--config",
                            help="Path to YAML configuration file",
                            default="configs/dwk.yaml",
                            type=str)
    return arg_parser


def vul_detect(config_path: str):
    filter_warnings()
    config = cast(DictConfig, OmegaConf.load(config_path))
    print_config(config, ["gnn", "classifier", "hyper_parameters"])
    seed_everything(config.seed, workers=True)

    vocab = Vocabulary.build_from_w2v(config.gnn.w2v_path)
    vocab_size = vocab.get_vocab_size()
    pad_idx = vocab.get_pad_id()

    # Init datamodule
    data_module = XFGDataModule(config, vocab)

    # Init model
    model = DeepWuKong(config, vocab, vocab_size, pad_idx)

    train(model, data_module, config)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    vul_detect(__args.config)
