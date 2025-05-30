from recbole.evaluator.base_metric import *
from recbole.evaluator.metrics import *
from recbole.evaluator.evaluator import *
from recbole.evaluator.register import *
from recbole.evaluator.collector import *
from recbole.evaluator.metrics_custom import CumulativeTailPercentage

from recbole.evaluator.metrics_custom import (
    CumulativeTailPercentage,
    CumulativeHeadPercentage
)

from recbole.evaluator import register as recbole_register
from recbole.utils import EvaluatorType
import logging

# Shorthand reference
metrics_to_register = {
    "cumulativetailpercentage": CumulativeTailPercentage,
    "cumulativeheadpercentage": CumulativeHeadPercentage
}

for name, metric_cls in metrics_to_register.items():
    if name not in recbole_register.metrics_dict:
        recbole_register.metrics_dict[name] = metric_cls
        recbole_register.metric_information[name] = metric_cls.metric_need
        recbole_register.metric_types[name] = metric_cls.metric_type

        # Only register as smaller-is-better if explicitly defined
        if getattr(metric_cls, "smaller", False):
            recbole_register.smaller_metrics.append(name)

        logging.getLogger().info(f"[Registered] {name} metric.")
