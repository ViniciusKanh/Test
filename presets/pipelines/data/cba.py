import marca

pipeline = marca.Pipeline(
    experiment_name="CBA",
    interest_measures_group=marca.interest_measures.InterestMeasuresGroup("CBA", ["confidence", "support"]),
    ranking_method=marca.rank.CBARank(),
    pruning_method=marca.prune.M1Prune(),
    classification_method=marca.classifier.OrdinalClassifier(),
)