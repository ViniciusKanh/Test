import marca

pipeline = marca.Pipeline(experiment_name="FeatureSelection",
                           feature_selection_method=[marca.feature_selection.FeatureSelection(),
                                                    marca.feature_selection.FeatureSelectionTipo2(),
                                                    marca.feature_selection.FeatureSelectionTipo3(),
                                                    marca.feature_selection.FeatureSelectionTipo4()
                                                    ],
                           ranking_method=[
                               marca.rank.CBARank(),
                           ],
                           pruning_method=[
                               marca.prune.M1Prune(),
                               marca.prune.M2Prune(),
                               marca.prune.M3Prune(),
                           ],
                           classification_method=[
                               marca.classifier.OrdinalClassifier(),
                           ],
                           add_cba=True,
                           )
