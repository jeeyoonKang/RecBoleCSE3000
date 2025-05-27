# -*- coding: utf-8 -*-
# @Time   : 2022/8/2
# @Author : Ayush Thakur
# @Email  : ayusht@wandb.com

r"""
recbole.utils.wandblogger
################################
"""
from collections import defaultdict

import numpy as np


class WandbLogger(object):
    """WandbLogger to log metrics to Weights and Biases."""

    def __init__(self, config):
        """
        Args:
            config (dict): A dictionary of parameters used by RecBole.
        """
        self.config = config
        self.log_wandb = config["log_wandb"]
        self._wandb = None
        self.setup()

    def setup(self):
        if self.log_wandb:
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                raise ImportError(
                    "To use the Weights and Biases Logger please install wandb."
                    "Run `pip install wandb` to install it."
                )

            # Initialize a W&B run
            if self._wandb.run is None:
                self._wandb.init(
                    project=self.config["wandb_project"],
                    name=self.config["wandb_run_name"],
                    config=self.config,
                )
                self.config["wandb_run_id"] = wandb.run.id
            else:
                print(self._wandb.run.id)
                print(self._wandb.run.name)
                print(self._wandb.run.dir)
            self._set_steps()

    def log_metrics(self, metrics, head="train", commit=True):
        if self.log_wandb:
            if head:
                metrics = self._add_head_to_metrics(metrics, head)
                self._wandb.log(metrics, commit=commit)
            else:
                self._wandb.log(metrics, commit=commit)

    def log_eval_metrics(self, metrics, head="eval"):
        if self.log_wandb:
            metrics = self._add_head_to_metrics(metrics, head)
            for k, v in metrics.items():
                self._wandb.run.summary[k] = v

    def _set_steps(self):
        self._wandb.define_metric("train/*", step_metric="train_step")
        self._wandb.define_metric("valid/*", step_metric="valid_step")

    def _add_head_to_metrics(self, metrics, head):
        head_metrics = dict()
        for k, v in metrics.items():
            if "_step" in k:
                head_metrics[k] = v
            else:
                head_metrics[f"{head}/{k}"] = v

        return head_metrics

    def log_group_eval(
        self, group_eval, group_name_mappings=None, group_eval_meta=None, step=None
    ):
        """
        Logs group-level evaluation metrics into one wandb.Table and logs summary statistics:
        - mean
        - std
        - absolute disparity (max - min)
        - coefficient of variation (std / mean)
        """
        if not self.log_wandb:
            return

        all_rows = []
        all_metrics = set()
        group_type_metrics = defaultdict(lambda: defaultdict(list))

        for group_type, group_metrics in group_eval.items():
            name_map = (
                group_name_mappings.get(group_type, {}) if group_name_mappings else {}
            )
            size_map = (
                group_eval_meta.get("user_group_sizes", {}).get(
                    group_type + "_group", {}
                )
                or group_eval_meta.get("user_group_sizes", {}).get(group_type, {})
                or group_eval_meta.get("item_group_sizes", {}).get(
                    group_type + "_group", {}
                )
                if group_eval_meta
                else {}
            )

            for group_id, metrics in group_metrics.items():
                group_name = name_map.get(group_id, str(group_id))
                group_size = size_map.get(str(group_id), -1)
                row = {
                    "GroupType": group_type,
                    "Group": group_name,
                    "GroupSize": group_size,
                    **metrics,
                }
                all_rows.append(row)

                for metric, value in metrics.items():
                    all_metrics.add(metric)
                    group_type_metrics[group_type][metric].append(value)

        if not all_rows:
            return

        all_metrics = sorted(all_metrics)
        columns = ["GroupType", "Group", "GroupSize"] + all_metrics
        data = [
            [row["GroupType"], row["Group"], row["GroupSize"]]
            + [row.get(m) for m in all_metrics]
            for row in all_rows
        ]

        table = self._wandb.Table(columns=columns, data=data)
        self._wandb.log({"group_eval/all_groups": table}, step=step)
        self.log_group_summary_table(group_type_metrics, step=step)

    def log_group_summary_table(self, group_type_metrics, step=None):
        """
        Logs group-level summary statistics (mean, std, disparity, coefficient of variation)
        as a wandb.Table for each metric across group types.
        """
        if not self.log_wandb:
            return

        summary_rows = []

        for group_type, metrics_dict in group_type_metrics.items():
            for metric, values in metrics_dict.items():
                values_np = np.array(values)
                mean = float(np.mean(values_np))
                std = float(np.std(values_np))
                disparity = float(np.max(values_np) - np.min(values_np))
                coeff_var = float(std / mean) if mean != 0 else 0.0

                summary_rows.append(
                    [group_type, metric, mean, std, disparity, coeff_var]
                )

        table = self._wandb.Table(
            columns=[
                "Group Type",
                "Metric",
                "Mean",
                "Std",
                "Abs Disparity",
                "Coeff Var",
            ],
            data=summary_rows,
        )

        self._wandb.log({"group_eval_summary": table}, step=step)

    def log_valid_score(self, valid_score):
        if self.log_wandb:
            self._wandb.log({"valid_score": valid_score})
