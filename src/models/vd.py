from torch import nn
from omegaconf import DictConfig
import torch
from src.datas.samples import XFGBatch
from typing import Dict
from pytorch_lightning import LightningModule
from src.models.modules.gnns import GraphConvEncoder, GatedGraphConvEncoder
from torch.optim import Adam, SGD, Adamax, RMSprop
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch.nn.functional as F
from src.metrics import Statistic
from torch_geometric.data import Batch
from src.vocabulary import Vocabulary


class DeepWuKong(LightningModule):
    r"""vulnerability detection model to detect vulnerability

    Args:
        config (DictConfig): configuration for the model
        vocabulary_size (int): the size of vacabulary
        pad_idx (int): the index of padding token
    """

    _optimizers = {
        "RMSprop": RMSprop,
        "Adam": Adam,
        "SGD": SGD,
        "Adamax": Adamax
    }

    _encoders = {
        "gcn": GraphConvEncoder,
        "ggnn": GatedGraphConvEncoder
    }

    def __init__(self, config: DictConfig, vocab: Vocabulary, vocabulary_size: int,
                 pad_idx: int):
        super().__init__()
        self.save_hyperparameters()
        self.__config = config
        hidden_size = config.classifier.hidden_size
        self.__graph_encoder = self._encoders[config.gnn.name](config.gnn, vocab, vocabulary_size,
                                                               pad_idx)
        # hidden layers
        layers = [
            nn.Linear(config.gnn.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.classifier.drop_out)
        ]
        if config.classifier.n_hidden_layers < 1:
            raise ValueError(
                f"Invalid layers number ({config.classifier.n_hidden_layers})")
        for _ in range(config.classifier.n_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(config.classifier.drop_out)
            ]
        self.__hidden_layers = nn.Sequential(*layers)
        self.__classifier = nn.Linear(hidden_size, config.classifier.n_classes)

    def forward(self, batch: Batch) -> torch.Tensor:
        """

        Args:
            batch (Batch): [n_XFG (Data)]

        Returns: classifier results: [n_method; n_classes]
        """
        # [n_XFG, hidden size]
        graph_hid = self.__graph_encoder(batch)
        hiddens = self.__hidden_layers(graph_hid)
        # [n_XFG; n_classes]
        return self.__classifier(hiddens)

    def _get_optimizer(self, name: str) -> torch.nn.Module:
        if name in self._optimizers:
            return self._optimizers[name]
        raise KeyError(f"Optimizer {name} is not supported")

    def configure_optimizers(self) -> Dict:
        parameters = [self.parameters()]
        optimizer = self._get_optimizer(
            self.__config.hyper_parameters.optimizer)(
            [{
                "params": p
            } for p in parameters],
            self.__config.hyper_parameters.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: self.__config.hyper_parameters.decay_gamma
                                    ** epoch)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _log_training_step(self, results: Dict):
        self.log_dict(results, on_step=True, on_epoch=False)

    def training_step(self, batch: XFGBatch,
                      batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"train_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="train")
            result.update(batch_metric)
            self._log_training_step(result)
            self.log("F1",
                     batch_metric["train_f1"],
                     prog_bar=True,
                     logger=False)
        return {"loss": loss, "statistic": statistic}

    def validation_step(self, batch: XFGBatch,
                        batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"val_loss": loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="val")
            result.update(batch_metric)
        return {"loss": loss, "statistic": statistic}

    def test_step(self, batch: XFGBatch,
                  batch_idx: int) -> torch.Tensor:  # type: ignore
        # [n_XFG; n_classes]
        logits = self(batch.graphs)
        loss = F.cross_entropy(logits, batch.labels)

        result: Dict = {"test_loss", loss}
        with torch.no_grad():
            _, preds = logits.max(dim=1)
            statistic = Statistic().calculate_statistic(
                batch.labels,
                preds,
                2,
            )
            batch_metric = statistic.calculate_metrics(group="test")
            result.update(batch_metric)

        return {"loss": loss, "statistic": statistic}

    # ========== EPOCH END ==========
    def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT,
                               step: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            losses = [
                so if isinstance(so, torch.Tensor) else so["loss"]
                for so in step_outputs
            ]
            mean_loss = torch.stack(losses).mean()
        return {f"{step}_loss": mean_loss}

    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, group: str):
        log = self._prepare_epoch_end_log(step_outputs, group)
        statistic = Statistic.union_statistics(
            [out["statistic"] for out in step_outputs])
        log.update(statistic.calculate_metrics(group))
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, training_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(training_step_output, "train")

    def validation_epoch_end(self, validation_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(validation_step_output, "val")

    def test_epoch_end(self, test_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(test_step_output, "test")
