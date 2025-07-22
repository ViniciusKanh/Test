import marca

pipeline = marca.Pipeline(
    experiment_name="CBA_ReliefF",
    feature_selection_method=[marca.feature_selection.ReliefFSelector(ratio=0.5)],
    ranking_method=marca.rank.CBARank(),
    pruning_method=marca.prune.M1Prune(),
    classification_method=marca.classifier.OrdinalClassifier(),
)
