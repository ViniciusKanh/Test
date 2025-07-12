import marca.interest_measures


class AlgorithmBuilder:
    def __init__(
        self,
        extraction,
        interest_measures,
        feature_selection,
        ranking,
        pruning,
        classification,
    ):
        self.name = None
        self.extraction = extraction
        self.interest_measures = interest_measures
        self.feature_selection = feature_selection
        rank = ranking
        if self.interest_measures is not None:
            rank.set_measures(self.interest_measures.get_measures())

        self.ranking = rank
        self.pruning = pruning

        self.special_cases()
        self.classification = classification

        self.steps = [
            step
            for step in [
                self.extraction,
                self.interest_measures,
                self.feature_selection,
                self.ranking,
                self.pruning,
                self.classification,
            ]
            if step is not None
        ]

        self.set_name("+".join([s.name for s in self.steps]))

    def special_cases(self):
        """Special cases for some methods to be used in the setup to create relations between steps of the setup"""
        pass

    def get_steps_names(self):
        return [s.name for s in self.steps]

    def get_steps_used(self):
        steps = []
        for step, method in self.get_params().items():
            if method is not None:
                steps.append(step)

        return steps

    def set_name(self, name):
        self.name = name

    def get_params(self):
        return {
            "extract": self.extraction,
            "interest_measures": self.interest_measures,
            "interest_measures_selection": self.feature_selection,
            "rank": self.ranking,
            "prune": self.pruning,
            "classifier": self.classification,
        }

    def __repr__(self):
        return self.name
