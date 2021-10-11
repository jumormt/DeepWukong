from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
from os import cpu_count
from os.path import join

import torch

from src.datas.samples import XFGBatch, XFGSample
from src.datas.datasets import XFGDataset
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
from src.vocabulary import Vocabulary


class XFGDataModule(LightningDataModule):
    def __init__(self, config: DictConfig, vocab: Vocabulary):
        super().__init__()
        self.__vocab = vocab
        self.__config = config
        self.__data_folder = join(config.data_folder, config.dataset.name)
        self.__n_workers = cpu_count(
        ) if self.__config.num_workers == -1 else self.__config.num_workers

    @staticmethod
    def collate_wrapper(batch: List[XFGSample]) -> XFGBatch:
        return XFGBatch(batch)

    def __create_dataset(self, data_path: str) -> Dataset:
        return XFGDataset(data_path, self.__config, self.__vocab)

    def train_dataloader(self) -> DataLoader:
        train_dataset_path = join(self.__data_folder, "train.json")
        train_dataset = self.__create_dataset(train_dataset_path)
        return DataLoader(
            train_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=self.__config.hyper_parameters.shuffle_data,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset_path = join(self.__data_folder, "val.json")
        val_dataset = self.__create_dataset(val_dataset_path)
        return DataLoader(
            val_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset_path = join(self.__data_folder, "test.json")
        test_dataset = self.__create_dataset(test_dataset_path)
        return DataLoader(
            test_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
        )

    def transfer_batch_to_device(
            self,
            batch: XFGBatch,
            device: Optional[torch.device] = None) -> XFGBatch:
        if device is not None:
            batch.move_to_device(device)
        return batch
