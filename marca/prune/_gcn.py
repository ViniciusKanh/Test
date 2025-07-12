from collections import Counter
import numpy as np
import pandas as pd
from marca.data_structures._cars import CARs
from sklearn.metrics import pairwise_distances
from ._prune import Prune
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

torch.manual_seed(1306)


class Net(torch.nn.Module):
    def __init__(self, numberAtt, numberNeurons, numberOfClasses):
        super(Net, self).__init__()
        self.conv1 = GCNConv(numberAtt, numberNeurons)  # dataset.num_node_features
        self.conv2 = GCNConv(numberNeurons, numberOfClasses)  # dataset.num_classes

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCNPrune(Prune):
    def __init__(
        self,
        distance_cut=0.1,
        class_cut=0.1,
        learningRate=0.001,
        numberNeurons=32,
        numberEpochs=400,
    ):
        super().__init__()
        self.name = "GCNPrune"
        self.distance_cut = distance_cut
        self.class_cut = class_cut
        self.learningRate = learningRate
        self.numberNeurons = numberNeurons
        self.numberEpochs = numberEpochs

        self.dataset = None
        self.rules = None
        self.interest_measures = None

        self.rank_together = False

    def calc_edges(self):
        self.interest_measures = self.rules.get_measures(
            self.interest_measures, normalized="rank"
        )
        # self.interest_measures = self.rules.get_measures(all_measures, normalized='rank')
        distances = pairwise_distances(
            self.interest_measures, self.interest_measures, metric="manhattan"
        ) / len(self.interest_measures[0])
        distances = distances <= self.distance_cut

        uper_edges = np.triu_indices(distances.shape[0])
        distances[uper_edges] = False
        np.fill_diagonal(distances, True)
        edges = np.array(np.where(distances)).T

        return edges

    def calc_classes(self):
        distances_to_max = (
            self.interest_measures.max(axis=0) - self.interest_measures
        ).sum(axis=1) / len(self.interest_measures)

        top = (
            pd.DataFrame(distances_to_max)
            .sort_values(by=0, ascending=True)[
                : round(len(distances_to_max) * self.class_cut)
            ]
            .index.values
        )
        worst = (
            pd.DataFrame(distances_to_max)
            .sort_values(by=0, ascending=False)[
                : round(len(distances_to_max) * self.class_cut)
            ]
            .index.values
        )

        train_test_index = np.array([False] * len(distances_to_max))
        train_test_index[top] = True
        train_test_index[worst] = True

        # Set class
        np.random.seed(1306)
        y = np.random.randint(2, size=distances_to_max.shape[0])
        y[top] = 1
        y[worst] = 0

        return self.interest_measures, train_test_index, y

    def run_gcn(self, attributes, classes, edges, train_test_mask):
        numberInstances = attributes.shape[0]
        number_classes = len(np.unique(classes))
        number_att = attributes.shape[1]

        train_mask = train_test_mask
        test_mask = ~train_test_mask

        edges = torch.tensor(edges).cuda().t().contiguous()
        x = torch.tensor(attributes).cuda()
        y = torch.tensor(classes).cuda()

        data = Data(
            x=x.float(),
            y=y,
            edge_index=edges,
            train_mask=train_mask,
            test_mask=test_mask,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Net(number_att, self.numberNeurons, number_classes).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.learningRate, weight_decay=5e-4
        )

        return self.train_gcn(model, optimizer, data)

    def train_gcn(self, model, optimizer, data):
        model.train()
        for epoch in range(self.numberEpochs):
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

        model.eval()

        _, pred = model(data).max(dim=1)

        results = model(data)
        rank_values = results.data[:, 0] / results.data[:, 1]

        return pred.detach().cpu().numpy(), rank_values

    def get_default_class(self, rules, data):
        index_covered = rules.get_cover_A(data)
        index_not_covered = list(
            set(range(0, len(data))).difference(index_covered.tolist())
        )

        data_not_covered = data[index_not_covered]

        if len(data_not_covered) == 0:
            return Counter(data[:, -1]).most_common(1)[0][0]

        else:
            return Counter(data_not_covered[:, -1]).most_common(1)[0][0]

    def __call__(self, X, y, rules):
        self.rules = rules
        # data = np.hstack((X.astype(float), y.astype(float).reshape(-1, 1)))

        if self.rank_together:
            pred = self.rules.prune_index
            final_classifier = CARs([self.rules[idx] for idx in np.where(pred == 1)[0]])

        else:
            edges = self.calc_edges()
            attributes, train_test_mask, classes = self.calc_classes()

            pred, rank_values = self.run_gcn(
                attributes, classes, edges, train_test_mask
            )
            final_classifier = CARs([self.rules[idx] for idx in np.where(pred == 1)[0]])

        default_rule = self.get_default_class(final_classifier, X.astype(float))
        return final_classifier, default_rule
