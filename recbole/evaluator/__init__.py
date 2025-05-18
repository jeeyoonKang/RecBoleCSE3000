from recbole.evaluator.base_metric import *
from recbole.evaluator.metrics import *
from recbole.evaluator.evaluator import *
from recbole.evaluator.register import *
from recbole.evaluator.collector import *
from recbole.evaluator.metrics_custom import CumulativeTailPercentage

from recbole.utils import EvaluatorType
import recbole.evaluator.metrics as recbole_metrics
import recbole.evaluator.register as recbole_register
import logging

# Register in metrics module namespace
recbole_metrics.CumulativeTailPercentage = CumulativeTailPercentage

# Manually register in RecBole's global metric registry if not already done
if "cumulativetailpercentage" not in recbole_register.metrics_dict:
    recbole_register.metrics_dict["cumulativetailpercentage"] = CumulativeTailPercentage
    recbole_register.metric_information["cumulativetailpercentage"] = (
        CumulativeTailPercentage.metric_need
    )
    recbole_register.metric_types["cumulativetailpercentage"] = (
        CumulativeTailPercentage.metric_type
    )

    if getattr(CumulativeTailPercentage, "smaller", False):
        recbole_register.smaller_metrics.append("cumulativetailpercentage")

    logging.getLogger().info("[Registered] CumulativeTailPercentage metric.")
