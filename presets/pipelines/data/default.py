import marca
from presets.interest_measures_groups import load_interest_measures_group

pipeline = marca.Pipeline(experiment_name="Pipeline4Test",
                           interest_measures_group=load_interest_measures_group('[TW]'),
                           ranking_method=[
                               marca.rank.BordaRank(method='mean'),
                               marca.rank.BordaRank(method='l2'),
                           ],
                           pruning_method=[
                               marca.prune.M1Prune(),
                               # marca.prune.DynamicPrune(),
                           ],
                           classification_method=[
                               marca.classifier.OrdinalClassifier(),
                               marca.classifier.VotingClassifier()
                           ],
                           add_cba=True,
                           )
