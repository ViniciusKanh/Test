from marca.associative_classifier import MARCA
from marca.classifier import OrdinalClassifier
from marca.interest_measures import InterestMeasuresGroup
from marca.prune import M1Prune
from marca.rank import InterestMeasureRank, TopsisRank
from marca.interest_measures_selection import Anova
from marca.extract import Apriori
from marca.datasets import load_iris


def test_marca_constructor():
    x_train, y_train, x_test, y_test = load_iris(fold=0)
    ext = Apriori(support=0, confidence=0, max_len=5)
    rules = ext(x_train, y_train)

    classifier = MARCA(
        rules=rules,
        interest_measures=["support", "confidence"],
        rank=InterestMeasureRank(measure="support"),
        prune=M1Prune(),
        classifier=OrdinalClassifier(),
    )

    classifier.fit(x_train, y_train)
    predict = classifier.predict(x_train)
    assert len(predict) == len(y_train)


def test_marca_FS():
    x_train, y_train, x_test, y_test = load_iris(fold=0)

    classifier = MARCA(
        extract=Apriori(support=0, confidence=0, max_len=5),
        interest_measures=["support", "confidence"],
        interest_measures_selection=Anova(),
        rank=TopsisRank(),
        prune=M1Prune(),
        classifier=OrdinalClassifier(),
    )

    classifier.fit(x_train, y_train)
    predict = classifier.predict(x_train)

    assert len(predict) == len(y_train)


def test_marca_constructor_set_params():
    x_train, y_train, x_test, y_test = load_iris(fold=0)
    ext = Apriori(support=0, confidence=0, max_len=5)
    rules = ext(x_train, y_train)

    classifier = MARCA(
        rules=rules,
        interest_measures=["support", "confidence"],
        rank=InterestMeasureRank(measure="support"),
        prune=M1Prune(),
        classifier=OrdinalClassifier(),
    )

    classifier.fit(x_train, y_train)
    predict = classifier.predict(x_train)

    classifier.set_params(
        **{
            "interest_measures": ["lift", "confidence"],
            "rank": InterestMeasureRank(measure="confidence"),
        }
    )
    classifier.fit(x_train, y_train)
    predict2 = classifier.predict(x_train)

    assert len(predict2) == len(y_train)


def test_marca_constructor_set_params2():
    x_train, y_train, x_test, y_test = load_iris(fold=0)
    ext = Apriori(support=0, confidence=0, max_len=5)
    rules = ext(x_train, y_train)

    classifier = MARCA(rules=rules)

    classifier.set_params(
        **{
            "interest_measures": InterestMeasuresGroup("Test", ["lift", "confidence"]),
            "rank": InterestMeasureRank(measure="confidence"),
            "prune": M1Prune(),
            "classifier": OrdinalClassifier(),
        }
    )
    classifier.fit(x_train, y_train)
    predict2 = classifier.predict(x_train)

    assert len(predict2) == len(y_train)
