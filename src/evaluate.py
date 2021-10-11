from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
import torch
from src.models.vd import DeepWuKong
from src.datas.datamodules import XFGDataModule
from src.utils import filter_warnings

def test(checkpoint_path: str, data_folder: str = None, batch_size: int = None):
    """

    test the trained model using specified files

    Args:
        checkpoint_path:
        data_folder:
        batch_size:

    Returns:

    """
    filter_warnings()
    model = DeepWuKong.load_from_checkpoint(checkpoint_path)
    config = model.hparams["config"]
    vocabulary = model.hparams["vocab"]
    if data_folder is not None:
        config.data_folder = data_folder
    if batch_size is not None:
        config.hyper_parameters.test_batch_size = batch_size
    data_module = XFGDataModule(config, vocabulary)
    seed_everything(config.seed)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(gpus=gpu)
    trainer.test(model, datamodule=data_module)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("--data-folder", type=str, default=None)
    arg_parser.add_argument("--batch-size", type=int, default=None)
    return arg_parser


if __name__ == '__main__':
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    test(__args.checkpoint, __args.data_folder, __args.batch_size)

