from recbole.evaluator.base_metric import AbstractMetric
from recbole.utils import EvaluatorType
import numpy as np
import logging


class CumulativeTailPercentage(AbstractMetric):
    """
    CumulativeTailPercentage computes the proportion of recommended items in the top-K list
    that fall into the bottom `tail_ratio` of total item interaction volume.

    This is fairness-aware: it defines tail items based on cumulative popularity mass.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.count_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]
        self.tail_ratio = config["tail_ratio"] if config["tail_ratio"] else 0.2
        self.logger = logging.getLogger()

    def used_info(self, dataobject):
        item_matrix = dataobject.get("rec.items")
        count_items = dataobject.get("data.count_items")
        return item_matrix.numpy(), dict(count_items)

    def get_tail_items(self, count_items):
        """
        Identify tail items based on cumulative popularity mass using `tail_ratio`.
        """
        sorted_items = sorted(
            count_items.items(), key=lambda x: (x[1], x[0])
        )  # ascending
        total = sum(cnt for _, cnt in sorted_items)
        threshold = self.tail_ratio * total

        cumulative = 0
        tail_items = set()
        for item, cnt in sorted_items:
            cumulative += cnt
            tail_items.add(item)
            if cumulative >= threshold:
                break

        self.logger.debug(f"Total interactions: {total}")
        self.logger.debug(f"Tail ratio threshold: {threshold}")
        self.logger.debug(f"Tail item count: {len(tail_items)}")
        return tail_items

    def get_tail_matrix(self, item_matrix, tail_items):
        return np.isin(item_matrix, list(tail_items)).astype(np.float32)

    def metric_info(self, values):
        return values[:, : max(self.topk)]

    def topk_result(self, metric, values):
        result = {}
        avg_result = values.mean(axis=0)
        for k in self.topk:
            result[f"{metric}@{k}"] = round(
                float(avg_result[k - 1]), self.decimal_place
            )
        return result

    def calculate_metric(self, dataobject):
        item_matrix, count_items = self.used_info(dataobject)
        tail_items = self.get_tail_items(count_items)
        tail_mask = self.get_tail_matrix(item_matrix, tail_items)
        metric_values = self.metric_info(tail_mask)
        return self.topk_result("cumulativetailpercentage", metric_values)
