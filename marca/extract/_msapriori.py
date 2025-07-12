import itertools
from collections import Counter
from marca.data_structures import CAR, CARs
import numpy as np
from sklearn.base import BaseEstimator
from marca.extract._extract import Extract


class MSApriori(BaseEstimator, Extract):
    def __init__(
            self,
            max_len=15,
            max_sup_diff=1,
            max_num_rules=20_000,
            B=0.25,
            name="",
            interest_measures=None,
    ):
        super().__init__()
        self.name = "MSApriori" + name
        self.interest_measures = interest_measures

        self.max_len = max_len
        self.max_sup_diff = max_sup_diff
        self.max_num_rules = max_num_rules

        self.itemsets = {}
        self.sup_count = {}
        self.candidates = {}
        self.number_instances = 0
        self.MIS = {}
        self.data_transformed = None
        self.must_have = []
        self.B = B

        self.cars = CARs()
        self.rules = {}

        self.supports = {}

    def set_support(self):
        for c in range(self.data_transformed.shape[1]):
            self.supports.update(dict(Counter(self.data_transformed[:, c])))

    def level1_candidate_gen(self):
        initial_values = [item for item, mis in sorted(list(self.MIS.items()), key=lambda x: (x[1], int(x[0])))]
        candidates_1 = []
        smallest_sup = None

        for item in initial_values:
            if smallest_sup:
                if (item in self.supports) and (self.supports[item] / self.number_instances >= smallest_sup):
                    candidates_1.append({'candidate': [item], 'count': self.supports[item]})

            elif (item in self.supports) and (self.supports[item] / self.number_instances >= self.MIS[item]):
                candidates_1.append({'candidate': [item], 'count': self.supports[item]})
                smallest_sup = self.MIS[item]

        itemsets_aux = []

        for value in candidates_1:
            value = value['candidate'][0]
            if self.supports[value] / self.number_instances >= self.MIS[value]:
                itemsets_aux.append([value])

        self.itemsets[1] = itemsets_aux
        self.candidates[1] = candidates_1

    def level2_candidate_gen(self):
        candidates_2 = []

        for idx, value in enumerate(self.candidates[1]):
            value = value['candidate'][0]
            if self.supports[value] / self.number_instances >= self.MIS[value]:
                for j in range(idx + 1, len(self.candidates[1])):
                    if ((self.supports[self.candidates[1][j]['candidate'][0]] / self.number_instances > self.MIS[
                        value]) and
                            (abs((self.supports[self.candidates[1][j]['candidate'][0]] / self.number_instances) - (
                                    self.supports[value] / self.number_instances)) <= self.max_sup_diff)):
                        candidates_2.append({'candidate': [value, self.candidates[1][j]['candidate'][0]], 'count': 0})

        for candidate in candidates_2:
            for t in self.data_transformed:
                if set(candidate['candidate']).issubset(t):
                    candidate['count'] += 1

        self.itemsets[2] = self.gen_itemsets_2(candidates_2)
        self.candidates[2] = candidates_2

    def leveln_candidate_gen(self, level):
        self.candidates[level] = []

        for index, f1 in enumerate(self.itemsets[level - 1]):
            for j, f2 in enumerate(self.itemsets[level - 1][index + 1:]):
                if ((set(f1[:-1]) == set(f2[:-1])) and
                        (abs(
                            (self.supports[f2[level - 2]] / self.number_instances) -
                            (self.supports[f1[level - 2]] / self.number_instances))
                         <= self.max_sup_diff)):
                    candidate = list(f1)
                    candidate.append(f2[level - 2])
                    delete = False
                    for s in list(itertools.combinations(candidate, level - 1)):
                        if candidate[0] in s or self.MIS[candidate[0]] == self.MIS[candidate[1]]:
                            if list(s) not in self.itemsets[level - 1]:
                                delete = True
                    if not delete:
                        self.candidates[level].append({'candidate': candidate, 'count': 0})

        self.gen_itemsets(level)

    def apply_constraints(self, level):
        self.rules[level] = []
        for f in self.itemsets[level]:
            if set(f).intersection(set(self.must_have)):
                self.rules[level].append(np.array(sorted(np.array(f))))

    def transform_to_sequential(self, X, y):
        """
        Transform the data to a sequential representation
        :param x:
        :return:
        """
        data_transformed = np.hstack((X, y.reshape(-1, 1))).astype(int)

        acc = 0
        len_ant = 0
        reverser = {}
        for c in range(data_transformed.shape[1]):
            acc += len_ant
            values = np.unique(data_transformed[:, c])
            reverser.update(dict(zip([x for x in values + acc], tuple(zip([c] * len(values), values)))))
            len_ant = len(values)
            data_transformed[:, c] = data_transformed[:, c] + acc

        self.reverser = reverser
        self.data_transformed = data_transformed
        self.number_instances = self.data_transformed.shape[0]
        self.num_attributes = self.data_transformed.shape[1]

    def set_MIS(self):
        """
        Set MIS for each item with 50% of the support
        :param x:
        :return:
        """
        t = self.data_transformed

        self.MIS = {}
        for c in range(t.shape[1]):
            val, count = np.unique(t[:, c], return_counts=True)
            self.MIS.update(dict(zip(val, self.B * (count / len(t)))))

    def gen_itemsets(self, level):
        for candidate in self.candidates[level]:
            sub_found = False
            sub_count = True

            for c1 in self.candidates[level - 1]:
                if set(candidate['candidate'][1:]) == set(c1['candidate']):
                    sub_found = True
                    if c1['count'] != 0:
                        sub_count = False
            if sub_count and not sub_found:
                self.candidates[level - 1].append({'candidate': candidate['candidate'][1:], 'count': 0})

            columns = [self.reverser[x][0] for x in candidate['candidate']]
            candidate['count'] += len(
                np.where((self.data_transformed[:, columns] == candidate['candidate']).all(axis=1))[0])
            if sub_count:
                self.candidates[level - 1][-1]['count'] += len(
                    np.where((self.data_transformed[:, columns[1:]] == candidate['candidate'][1:]).all(axis=1))[0])

        self.itemsets[level] = []
        for candidate in self.candidates[level]:
            if (candidate['count'] / self.number_instances) >= self.MIS[candidate['candidate'][0]]:
                self.itemsets[level].append(candidate['candidate'])

    def gen_itemsets_2(self, candidates):
        """
        Filter candidates by MIS
        :param candidates:
        :return:
        """
        itemsets = []
        for candidate in candidates:
            if candidate['count'] / self.number_instances >= self.MIS[candidate['candidate'][0]]:
                itemsets.append(candidate['candidate'])

        return itemsets

    def set_must_have(self):
        self.must_have = list(np.unique(self.data_transformed[:, -1]))

    def get_antecedent_support(self, item, level):
        for c in self.candidates[level - 1]:
            if set(c['candidate']) == set(item):
                return c['count']

    def get_consequent_support(self, item):
        return self.supports[item]

    def get_ant_cons_support(self, item, level):
        for c in self.candidates[level]:
            if set(c['candidate']) == set(item):
                return c['count']

    def generate_rules(self, level):
        self.apply_constraints(level)

        for rule in self.rules[level]:
            car = np.array([np.nan] * self.num_attributes)
            for item in rule:
                col, value = self.reverser[item]
                car[col] = value

            antecedent, consequent = rule[:-1], rule[-1]
            f_lhs = self.get_antecedent_support(antecedent, level) / self.number_instances
            f_rhs = self.get_consequent_support(consequent) / self.number_instances
            f_lhs_rhs = self.get_ant_cons_support(rule, level) / self.number_instances

            self.cars.append(CAR.from_numpy(rule=car, sup_rule=f_lhs_rhs, sup_antecedent=f_lhs, sup_consequent=f_rhs, n_transactions=self.number_instances))

    def __call__(self, X, y):
        self.transform_to_sequential(X, y)
        self.set_must_have()
        self.set_MIS()
        self.set_support()

        self.level1_candidate_gen()
        self.level2_candidate_gen()
        self.generate_rules(level=2)
        # print(len(self.cars))

        level = 3
        while (self.candidates[level - 1]) and (level <= self.max_len) and (len(self.cars) < self.max_num_rules):
            # print(f'--- gen level {level} ----')
            self.leveln_candidate_gen(level)
            self.generate_rules(level=level)
            # print(len(self.cars))
            level += 1

        return self.cars
