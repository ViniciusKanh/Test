from sklearn.base import BaseEstimator
from ._associative_classifier import AssociativeClassifier
import time
from marca.interest_measures import InterestMeasuresGroup, InterestMeasures


class ModularClassifier(BaseEstimator, AssociativeClassifier):
    def __init__(
            self,
            extract=None,
            interest_measures=None,
            interest_measures_selection=None,
            rank=None,
            prune=None,
            classifier=None,
            rules=None,
            reuse_params=True,
    ):
        """

        Parameters
        ----------
        extract
        interest_measures
        interest_measures_selection
        rank
        prune
        classifier
        rules
        reuse_params
        """
        super().__init__()

        # Attributes of the classifier
        self.interest_measures = None
        self.selected_interest_measures = None
        self.rules_sorted = None
        self.rules_pruned = None
        self.default_class = None

        # Attributes of last fit
        self.X = None
        self.y = None
        self.rules = rules

        # Time of each step of the classifier
        self.time = {
            "extract": 0.0,
            "interest_measures": 0.0,
            "interest_measures_selection": 0.0,
            "rank": 0.0,
            "prune": 0.0,
            "classifier": 0.0,
        }

        # Steps of the classifier
        self.steps = [
            "extract",
            "interest_measures",
            "interest_measures_selection",
            "rank",
            "prune",
            "classifier",
        ]

        self.extract = extract

        self._check_interest_measures(interest_measures)
        self.interest_measures_selection = interest_measures_selection
        self.rank = rank
        self.prune = prune
        self.classifier = classifier

        # Parameters to reuse of classifier for optimization
        self.reuse_params = reuse_params
        self.need_refit = {
            "extract": True,
            "interest_measures": True,
            "interest_measures_selection": True,
            "rank": True,
            "prune": True,
            "classifier": True,
        }

        if extract is None:
            self.need_refit["extract"] = False
        if interest_measures_selection is None:
            self.need_refit["interest_measures_selection"] = False

    def _check_interest_measures(self, interest_measures):
        if isinstance(interest_measures, InterestMeasuresGroup):
            self.interest_measures = interest_measures
        else:
            if interest_measures is None:
                self.interest_measures = InterestMeasuresGroup("AllMeasures", measures=InterestMeasures.measures_available.keys())
            else:
                self.interest_measures = InterestMeasuresGroup(
                    "Default", interest_measures
                )

    def _map_return(self, step, return_value):
        if step == "extract":
            self.rules = return_value

        if step == "interest_measures":
            self.interest_measures = return_value

        elif step == "interest_measures_selection":
            self.selected_interest_measures = return_value

        elif step == "rank":
            self.rules_sorted = return_value

        elif step == "prune":
            self.rules_pruned = return_value[0]
            self.default_class = return_value[1]

    def _get_param(self, step):
        if step == "extract":
            return {"X": self.X, "y": self.y}

        elif step == "interest_measures":
            return {}

        elif step == "interest_measures_selection":
            return {"rules": self.rules, "measures": self.interest_measures}

        elif step == "rank":
            if self.selected_interest_measures is not None:
                return {
                    "x": self.X,
                    "y": self.y,
                    "rules": self.rules,
                    "measures": self.selected_interest_measures,
                }
            else:
                return {"x": self.X,
                        "y": self.y,
                        "rules": self.rules, "measures": self.interest_measures}

        elif step == "prune":
            return {"X": self.X, "y": self.y, "rules": self.rules_sorted}

        elif step == "classifier":
            return {"rules": self.rules_pruned, "default_class": self.default_class}

    def _reset_steps(self):
        for step in self.steps:
            self.need_refit[step] = False

    def _reuse_params(self, params):
        self._reset_steps()
        if "extract" in params:
            if self.extract != params["extract"]:
                self.extract = params["extract"]
                self.need_refit['extract'] = True
                # if self.extract is not None:
                #     self.rules = params["extract"]

        if "interest_measures" in params:
            if self.interest_measures != params["interest_measures"]:
                self._check_interest_measures(params["interest_measures"])
                self.need_refit["interest_measures"] = True
                self.need_refit["rank"] = True
                self.need_refit["prune"] = True
                self.need_refit["classifier"] = True

                if "interest_measures_selection" in params:
                    self.need_refit["interest_measures_selection"] = True
                else:
                    self.need_refit["interest_measures_selection"] = False

        if "interest_measures_selection" in params:
            if params["interest_measures_selection"] is not None:
                if self.interest_measures_selection != params["interest_measures_selection"]:
                    self.interest_measures_selection = params["interest_measures_selection"]
                    self.need_refit["interest_measures_selection"] = True
                    self.need_refit["rank"] = True
                    self.need_refit["prune"] = True
                    self.need_refit["classifier"] = True

            else:
                self.need_refit["interest_measures_selection"] = False

        if "rank" in params:
            if self.rank != params["rank"]:
                self.rank = params["rank"]
                self.need_refit["rank"] = True
                self.need_refit["prune"] = True
                self.need_refit["classifier"] = True

        if "prune" in params:
            if self.prune != params["prune"]:
                self.prune = params["prune"]
                self.need_refit["prune"] = True
                self.need_refit["classifier"] = True

        if "classifier" in params:
            if self.classifier != params["classifier"]:
                self.classifier = params["classifier"]
                self.need_refit["classifier"] = True

    def set_params(self, **params):
        if not self.reuse_params:
            super().set_params(**params)
        else:
            self._reuse_params(params)

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return:
        """
        self.X = X
        self.y = y

        for step in self.steps:
            if self.need_refit[step]:
                time_init = time.time()
                value_return = getattr(self, step)(**self._get_param(step))
                self._map_return(step, value_return)

                time_end = time.time()
                self.time[step] = time_end-time_init
