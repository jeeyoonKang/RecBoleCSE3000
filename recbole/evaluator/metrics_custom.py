from recbole.evaluator.base_metric import AbstractMetric
from recbole.utils import EvaluatorType
import numpy as np
import logging


class CumulativeTailPercentage(AbstractMetric):
    """
    CumulativeTailPercentage computes the average proportion of tail items—defined by cumulative
    popularity mass—recommended in the top-K list.

    Tail items are identified based on the lowest cumulative interaction volume, such that they
    collectively account for the bottom `tail_ratio` (e.g., 20%) of all item interactions.
    This makes the metric sensitive to popularity imbalance and suitable for fairness evaluation
    in long-tail recommendation.

    For each user, it computes the fraction of tail items in the top-K list at each K, and
    averages this across all users.
    """

    # Define this as a ranking metric
    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.count_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]  # List of cutoff points (e.g., [5, 10, 20])
        self.tail_ratio = config["tail_ratio"] if config["tail_ratio"] else 0.2  # Ratio of cumulative interaction mass defining the tail
        self.logger = logging.getLogger()
        self.show_progress = config["show_progress"]

    def used_info(self, dataobject):
        """
        Extract the top-K recommendation lists and item interaction counts from the evaluation data.

        Returns:
            item_matrix (ndarray): shape [num_users, top_k], recommended item IDs.
            count_items (dict): item_id -> interaction count in training data.
        """
        item_matrix = dataobject.get("rec.items")
        count_items = dataobject.get("data.count_items")
        return item_matrix.numpy(), dict(count_items)

    def get_tail_items(self, count_items):
        """
        Determine the set of tail items based on cumulative interaction mass.

        Tail items are the least popular items that together account for up to `tail_ratio`
        of the total interactions across all items.

        Args:
            count_items (dict): item_id -> interaction count.

        Returns:
            set: IDs of items considered as tail.
        """
        # Sort items by interaction count (ascending), then by item ID
        sorted_items = sorted(count_items.items(), key=lambda x: (x[1], x[0]))
        total = sum(cnt for _, cnt in sorted_items)
        threshold = self.tail_ratio * total  # Cumulative mass threshold

        cumulative = 0
        tail_items = set()
        for item, cnt in sorted_items:
            cumulative += cnt
            tail_items.add(item)
            if cumulative >= threshold:
                break


        # Debug logging
        self.logger.debug(f"Total interactions: {total}")
        self.logger.debug(f"Tail ratio threshold: {threshold}")
        self.logger.debug(f"Tail item count: {len(tail_items)}")
        return tail_items


    def get_tail_matrix(self, item_matrix, tail_items):
        """
        Create a binary matrix indicating which recommended items are in the tail.

        Args:
            item_matrix (ndarray): [num_users, top_k] recommended items.
            tail_items (set): Set of tail item IDs.

        Returns:
            ndarray: binary matrix [num_users, top_k], where 1 indicates a tail item.
        """
        return np.isin(item_matrix, list(tail_items)).astype(np.float32)



    def metric_info(self, values):
        """
        Compute the cumulative average proportion of tail items up to each top-k.

        Args:
            values (ndarray): binary matrix indicating presence of tail items.

        Returns:
            ndarray: [num_users, top_k], cumulative tail ratio for each user at each k.
        """
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, values):
        """
        Compute the final metric values for each top-k.

        Args:
            metric (str): Metric name prefix (e.g., "cumulativetailpercentage").
            values (ndarray): [num_users, top_k] cumulative values per user.

        Returns:
            dict: {f"{metric}@k": score}, average score over all users for each k.
        """
        result = {}
        avg_result = values.mean(axis=0)  # Average across all users
        for k in self.topk:
            result[f"{metric}@{k}"] = round(float(avg_result[k - 1]), self.decimal_place)
        return result

    def calculate_metric(self, dataobject):
        """
        Main evaluation pipeline:
        1. Extract recommendation and interaction data.
        2. Identify tail items based on cumulative popularity.
        3. Generate binary matrix marking tail items in recommendations.
        4. Compute cumulative average tail ratios up to each top-k.
        5. Return average scores across users for all configured top-k values.

        Returns:
            dict: Final metric values keyed by top-k.
        """
        item_matrix, count_items = self.used_info(dataobject)
        tail_items = self.get_tail_items(count_items)

        tail_mask = self.get_tail_matrix(item_matrix, tail_items)
        tail_values = self.metric_info(tail_mask)
        result = self.topk_result("cumulativetailpercentage", tail_values)
        return result





class CumulativeHeadPercentage(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ["rec.items", "data.count_items"]

    def __init__(self, config):
        super().__init__(config)
        self.topk = config["topk"]
        self.tail_ratio = config["tail_ratio"] if config[
            "tail_ratio"] else 0.2  # Ratio of cumulative interaction mass defining the tail
        self.logger = logging.getLogger()

    def used_info(self, dataobject):
        item_matrix = dataobject.get("rec.items")
        count_items = dataobject.get("data.count_items")
        return item_matrix.numpy(), dict(count_items)

    def get_head_items(self, count_items):
        sorted_items = sorted(count_items.items(), key=lambda x: (x[1], x[0]))
        total = sum(cnt for _, cnt in sorted_items)
        threshold = self.tail_ratio * total

        head_items, cumulative = set(), 0
        for item, cnt in reversed(sorted_items):
            cumulative += cnt
            head_items.add(item)
            if cumulative >= threshold:
                break

        return head_items

    def get_group_mask(self, item_matrix, item_set):
        return np.isin(item_matrix, list(item_set)).astype(np.float32)

    def metric_info(self, values):
        return values.cumsum(axis=1) / np.arange(1, values.shape[1] + 1)

    def topk_result(self, metric, values):
        result = {}
        avg_result = values.mean(axis=0)
        for k in self.topk:
            result[f"{metric}@{k}"] = round(float(avg_result[k - 1]), self.decimal_place)
        return result

    def calculate_metric(self, dataobject):
        item_matrix, count_items = self.used_info(dataobject)
        head_items = self.get_head_items(count_items)
        head_mask = self.get_group_mask(item_matrix, head_items)
        head_values = self.metric_info(head_mask)
        return self.topk_result("headpercentage", head_values)


