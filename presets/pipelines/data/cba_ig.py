import marca

pipeline = marca.Pipeline(
    experiment_name="CBA_IG",
    feature_selection_method=[marca.feature_selection.IGSelector(ratio=0.5)],
    ranking_method=marca.rank.CBARank(),
    pruning_method=marca.prune.M1Prune(),
    classification_method=marca.classifier.OrdinalClassifier(),
)
