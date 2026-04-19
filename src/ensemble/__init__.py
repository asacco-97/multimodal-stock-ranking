"""
Ensemble modeling module for stock ranking.

Public API:
    StockRankerEnsemble  - Combine multiple rankers via rank-average or stacking.
    HMMRegimeDetector    - Gaussian HMM market regime classifier.
    anchored_expanding_cv - Time-series CV with anchored expanding windows.
    tune_hyperparameters  - Two-phase hyperparameter search.
    BaseRanker, XGBRanker, MLPRanker, ElasticNetRanker - Individual rankers.
"""

from .cv import anchored_expanding_cv, tune_hyperparameters
from .ensemble import StockRankerEnsemble
from .rankers import BaseRanker, ElasticNetRanker, MLPRanker, XGBRanker
from .regime import HMMRegimeDetector

__all__ = [
    "StockRankerEnsemble",
    "HMMRegimeDetector",
    "anchored_expanding_cv",
    "tune_hyperparameters",
    "BaseRanker",
    "XGBRanker",
    "MLPRanker",
    "ElasticNetRanker",
]
