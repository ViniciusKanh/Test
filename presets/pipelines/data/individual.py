import marca


class SpecialBuilder(marca.pipeline.AlgorithmBuilder):
    def special_cases(self):
        if isinstance(self.ranking, marca.rank.InterestMeasureRank):
            self.ranking.set_measure(self.interest_measures)

pipeline = marca.Pipeline(experiment_name="InterestMeasuresComSup",
                           # interest_measures_group=individual_measures,
                           ranking_method=[
                               marca.rank.InterestMeasureRank(),
                           ],
                           pruning_method=[
                               marca.prune.M1Prune(),
                           ],
                           classification_method=[
                               marca.classifier.OrdinalClassifier(),
                           ],
                           add_cba=True,
                           builder=SpecialBuilder
                           )
