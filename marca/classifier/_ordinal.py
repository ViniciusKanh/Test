from ._classifier import Classifier


class OrdinalClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.name = "OrdinalClassifier"
        self.rules = None
        self.default_class = None

    def __call__(self, rules, default_class):
        self.rules = rules
        self.default_class = default_class

    def first_match(self, instance):
        for rule in self.rules:
            if rule.match(instance):
                return rule.consequent

        return self.default_class

    def predict(self, data):
        data = data.astype("float")
        if len(data) > 1:
            predicts = []

            for instance in data:
                classe = self.first_match(instance)
                predicts.append(classe)

            return predicts

        else:
            return self.first_match(data)
