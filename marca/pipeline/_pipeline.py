from copy import deepcopy
from marca.interest_measures import InterestMeasuresGroup
from marca.rank import CBARank
from marca.prune import M1Prune
from marca.classifier import OrdinalClassifier
from ._builder import AlgorithmBuilder


class Pipeline:
    def __init__(
        self,
        experiment_name,
        extraction_method=None,
        interest_measures_group=None,
        feature_selection_method=None,
        ranking_method=None,
        pruning_method=None,
        classification_method=None,
        add_cba=False,
        builder=AlgorithmBuilder,
    ):
        self.builder = builder
        self.experiment_name = experiment_name

        self.steps = dict()
        self.steps["extraction_method"] = extraction_method
        self.steps["interest_measures_group"] = interest_measures_group
        self.steps["feature_selection_method"] = feature_selection_method
        self.steps["ranking_method"] = ranking_method
        self.steps["pruning_method"] = pruning_method
        self.steps["classification_method"] = classification_method

        self.cba_enable = add_cba

        # pops = []
        for step, method in self.steps.items():
            self.steps[step] = self._validate_method(method)

    def _validate_method(self, method):
        if isinstance(method, list):
            return method
        else:
            return [method]

    def __len__(self):
        aux = 1
        for s in self.steps.values():
            if len(s) == 0:
                aux = aux
            else:
                aux *= len(s)

        return aux

    def get(self):
        pipelines = []

        if self.cba_enable:
            pipelines.append(
                AlgorithmBuilder(
                    extraction=None,
                    interest_measures=InterestMeasuresGroup("CBA", ["confidence", "support"]),
                    feature_selection=None,
                    ranking=CBARank(),
                    pruning=M1Prune(),
                    classification=OrdinalClassifier(),
                )
            )

        for extraction in self.steps["extraction_method"]:
            extract = deepcopy(extraction)
            for interest_measures in self.steps["interest_measures_group"]:
                for feature_selection in self.steps["feature_selection_method"]:
                    for ranking in self.steps["ranking_method"]:
                        rank = deepcopy(ranking)
                        for pruning in self.steps["pruning_method"]:
                            prune = deepcopy(pruning)
                            for classification in self.steps["classification_method"]:
                                pipelines.append(
                                    self.builder(
                                        extraction=extract,
                                        interest_measures=interest_measures,
                                        feature_selection=feature_selection,
                                        ranking=rank,
                                        pruning=prune,
                                        classification=classification,
                                    )
                                )

        return pipelines