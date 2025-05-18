from recbole.evaluator.base_metric import AbstractMetric
from recbole.utils import EvaluatorType
import numpy as np
import logging
class CumulativeTailPercentage(AbstractMetric):
    """
    Metric: CumulativeTailPercentage

    Measures the cumulative proportion of recommended items that fall into the 'tail' of the item popularity distribution.
    Tail items are defined as those outside the top-N% of interactions (complement of head_ratio).
    This is a fairness-aware metric that helps assess how well a model surfaces long-tail content.
    """

    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.count_items"]
    smaller = False  # Higher value is better (more tail coverage)

    def __init__(self, config):
        """
        Initialize metric with configuration.
        """
        super().__init__(config)
        self.topk = config["topk"]
        self.head_ratio = config["head_ratio"] if "head_ratio" in config else 1 - config["tail_ratio"]
        self._cached_tail_items = None
        self.logger = logging.getLogger()

    def used_info(self, dataobject):
        """
        Extract item recommendation matrix and item popularity counts.
        """
        item_matrix = dataobject.get("rec.items")
        count_items = dataobject.get("data.count_items")
        return item_matrix.numpy(), dict(count_items)

    def get_tail_items(self, count_items):
        """
        Identify tail items by removing the head items that make up top `head_ratio` of total interactions.
        """
        if self._cached_tail_items is not None:
            return self._cached_tail_items

        sorted_items = sorted(count_items.items(), key=lambda x: x[1], reverse=True)
        total = sum(cnt for _, cnt in sorted_items)

        self.logger.debug(f"Total interactions: {total}")
        self.logger.debug(f"Total unique items: {len(count_items)}")
        self.logger.debug(f"Head ratio: {self.head_ratio} (top {self.head_ratio:.0%} of interactions)")

        cumulative = 0
        head_items = set()
        for item, cnt in sorted_items:
            cumulative += cnt
            head_items.add(item)
            if cumulative >= self.head_ratio * total:
                break

        tail_items = set(count_items.keys()) - head_items

        self.logger.debug(f"Head item count: {len(head_items)}")
        self.logger.debug(f"Tail item count: {len(tail_items)}")
        self.logger.debug(f"Sample tail item IDs: {list(tail_items)[:10]}")

        self._cached_tail_items = tail_items
        return tail_items

    def get_tail_matrix(self, item_matrix, tail_items):
        """
        Create a binary matrix indicating whether each recommended item is a tail item.
        """
        return np.isin(item_matrix, list(tail_items)).astype(np.float32)

    def metric_info(self, values):
        """
        Calculate cumulative precision across top-k positions.
        """
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, values):
        """
        Aggregate and format the metric result for each top-k level.
        """
        result = {}
        avg_result = values.mean(axis=0)
        for k in self.topk:
            result[f"{metric}@{k}"] = round(avg_result[k - 1], self.decimal_place)
        return result

    def calculate_metric(self, dataobject):
        """
        Full computation pipeline for CumulativeTailPercentage.
        """
        item_matrix, count_items = self.used_info(dataobject)
        tail_items = self.get_tail_items(count_items)
        tail_mask = self.get_tail_matrix(item_matrix, tail_items)
        metric_values = self.metric_info(tail_mask)
        return self.topk_result("cumulativetailpercentage", metric_values)
