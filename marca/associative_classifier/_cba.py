from sklearn.base import BaseEstimator
from ._associative_classifier import AssociativeClassifier
from ..extract import Apriori
from ..rank import CBARank
from ..prune import M1Prune
from ..classifier import OrdinalClassifier


class CBA(BaseEstimator, AssociativeClassifier):
    def __init__(
            self,
            support=0,
            confidence=0,
            max_len=5,
    ):
        super().__init__(
            extract=Apriori(support=support, confidence=confidence, max_len=max_len),
            interest_measures_selection=CBARank(),
            rank=CBARank(),
            prune=M1Prune(),
            classifier=OrdinalClassifier()
        )
    
    def fit(self, X, y):
        rules = self.extract(X, y)
        ranked_rules = self.rank(x=X, y=y, rules=rules)
        self.rules_pruned, self.default_class = self.prune(x=X, y=y, rules=ranked_rules)
        self.classifier(self.rules_pruned, self.default_class)
        return self
    
    def predict(self, X):
        return self.classifier.predict(X)
    
    def score(self, X, y):
        return self.classifier.score(X, y)
    
    def get_params(self, deep=True):
        return {
            "support": self.extract.support,
            "confidence": self.extract.confidence,
            "max_len": self.extract.max_len,
        }
    
    def set_params(self, **params):
        self.extract.support = params["support"]
        self.extract.confidence = params["confidence"]
        self.extract.max_len = params["max_len"]
