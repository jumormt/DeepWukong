from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class Statistic:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def update(self, other_statistic: "Statistic"):
        self.true_positive += other_statistic.true_positive
        self.false_positive += other_statistic.false_positive
        self.false_negative += other_statistic.false_negative
        self.true_negative += other_statistic.true_negative

    def calculate_metrics(self, group: Optional[str] = None) -> Dict[str, int]:
        precision, recall, f1, fpr, acc = 0, 0, 0, 0, 0
        acc = (self.true_negative + self.true_positive) / (
                self.true_positive + self.true_negative + self.false_positive +
                self.false_negative)
        if self.true_positive + self.false_positive > 0:
            precision = self.true_positive / (self.true_positive +
                                              self.false_positive)
        if self.false_positive + self.true_negative > 0:
            fpr = self.false_positive / (self.false_positive +
                                         self.true_negative)
        if self.true_positive + self.false_negative > 0:
            recall = self.true_positive / (self.true_positive +
                                           self.false_negative)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        metrics_dict = {
            "fpr": fpr.item() if type(fpr) == torch.Tensor else fpr,
            "precision":
                precision.item() if type(precision) == torch.Tensor else precision,
            "recall":
                recall.item() if type(recall) == torch.Tensor else recall,
            "accuracy": acc.item() if type(acc) == torch.Tensor else acc,
            "f1": f1.item() if type(f1) == torch.Tensor else f1
        }
        if group is not None:
            for key in list(metrics_dict.keys()):
                metrics_dict[f"{group}_{key}"] = metrics_dict.pop(key)
        return metrics_dict

    @staticmethod
    def calculate_statistic(labels: torch.Tensor, preds: torch.Tensor,
                            nb_classes: int) -> "Statistic":
        """Calculate statistic for ground truth and predicted batches of labels.
        :param labels: ground truth labels
        :param preds: predicted labels
        :param skip: list of subtokens ids that should be ignored
        :return: dataclass with calculated statistic
        """
        statistic = Statistic()
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        statistic.true_negative, statistic.false_positive, statistic.false_negative, statistic.true_positive = \
        confusion_matrix[
            0,
            0], confusion_matrix[0,
                                 1], confusion_matrix[1,
                                                      0], confusion_matrix[1,
                                                                           1]
        return statistic

    @staticmethod
    def union_statistics(stats: List["Statistic"]) -> "Statistic":
        union_statistic = Statistic()
        for stat in stats:
            union_statistic.update(stat)
        return union_statistic
