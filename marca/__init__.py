__version__ = "0.1.0"

import marca.classifier
import marca.prune
import marca.rank
import marca.extract
import marca.associative_classifier
import marca.data_structures
import marca.interest_measures
from marca.associative_classifier import ModularClassifier
from marca.data_structures import CAR as CAR
from marca.data_structures import CARs
from marca.pipeline import Pipeline
from marca.feature_selection import IGSelector, ReliefFSelector, EnsembleSelector
__all__ = [
    "marca",
    "classifier",
    "prune",
    "rank",
    "extract",
    "associative_classifier",
    "data_structures",
    "interest_measures",
    "feature_selection",
    "CAR",
    "CARs",
    "Pipeline"
]
