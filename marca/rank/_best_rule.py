from ._rank import Rank
from marca.interest_measures import InterestMeasuresGroup, InterestMeasures
import pandas as pd
import numpy as np


class BestRuleRank(Rank):
    def __init__(self, measures=None, name=""):
        super().__init__()
        self.name = "BestRule" + name
        self.measures = measures

    @Rank.rankmethod
    def __call__(self, x, y, rules, measures):
        # mos = rules.get_measures(measures=measures, normalized="rank")

        all = InterestMeasuresGroup('Todas', list(
            InterestMeasures.measures_available.keys()))
        mos = rules.get_measures(all)
        mos = pd.DataFrame(mos, columns=all.measures)

        contagem_melhores_por_mos = pd.DataFrame([mos[m] == mos[m].max() for m in mos.columns]).T.sum(axis=0)
        mos_escolhidas = contagem_melhores_por_mos.index[np.where(pd.DataFrame(contagem_melhores_por_mos) < 100)[0]]
        print(mos_escolhidas)
        index_final = []
        soma = []
        new_mos = mos
        for _ in range(min(1000, len(new_mos))):
            contagem = pd.DataFrame([new_mos[m] == new_mos[m].max() for m in new_mos.columns]).T.sum(
                axis=1).sort_values(ascending=False)
            index_escolhido = contagem.index[0]
            value = contagem.values[0]
            index_final.append(index_escolhido)
            soma.append(value)

            new_mos = mos.drop(index=index_final)

        return 1/np.argsort(index_final)
