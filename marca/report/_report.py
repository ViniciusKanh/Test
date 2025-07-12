import pickle
import numpy as np
import time
import pandas as pd
from rich.table import Table
import os
import sys

sys.path.insert(0, '../../../marca')
def drop_cba_redudant(df):
    return df.drop(
        index=df.loc[
            (df.level_0 != "CBA")
            & (df.level_1 == "CBARank")
            & (df.level_2 == "M1Prune")
            & (df.level_3 == "OrdinalClassifier")
        ].index
    )


def process_reports(report_name):
    report = load_reports(report_name, "All")
    for measure in ["F1", "Time", "Overlap", "Length", "Size"]:
        aux = report[measure]
        aux.columns = pd.MultiIndex.from_arrays(
            np.array([s.split("+") for s in aux.columns]).T
        )
        report[measure] = report[measure].T.reset_index()
        report[measure] = drop_cba_redudant(report[measure])

    pickle.dump(report, open(f"reports/processed/{report_name}.pkl", "wb"))


def concat_reports(experiment_name1, experiment_name2):
    os.mkdir(f"reports/raw/{experiment_name1}_{experiment_name2}")

    for dataset_name in os.listdir(f"reports/raw/{experiment_name1}"):
        if dataset_name.endswith(".pkl"):
            reports = pickle.load(open(f"reports/raw/{experiment_name1}/{dataset_name}", "rb"))
            reports.update(pickle.load(open(f"reports/raw/{experiment_name2}/{dataset_name}", "rb")))

            pickle.dump(
                reports,
                open(f"reports/raw/{experiment_name1}_{experiment_name2}/{dataset_name}", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def load_reports(experiment_name, measure=None):
    """
    Load all reports inside folder experiment_name and return dataframe of reports
    Parameters
    ----------
    experiment_name: str
    measure: str

    Returns
    -------
    dict
    """
    reports = {}
    for f in os.listdir(f"reports/raw/{experiment_name}"):
        if f.endswith(".pkl"):
            with open(f"reports/raw/{experiment_name}/{f}", "rb") as file:
                reports.update({f[:-4]: pickle.load(file)})

    if measure is None:
        return reports

    else:
        aux = None
        if measure.capitalize() == "F1":
            return pd.DataFrame({dataset: {exp: np.mean([x .f1 for x in reports[dataset][exp]['report']]) for exp in reports[dataset].keys()} for dataset in reports.keys()}).T

        if measure.capitalize() == "Time":
            aux = {
                dataset: {
                    k: np.mean([x.time_calc for x in v]) for k, v in experiments.items()
                }
                for dataset, experiments in reports.items()
            }

        if measure.capitalize() == "Size":
            aux = {
                dataset: {
                    k: np.mean([len(x.rules_pruned) for x in v["report"]])
                    for k, v in experiments.items()
                }
                for dataset, experiments in reports.items()
            }

        if measure.capitalize() == "Length":
            aux = {
                dataset: {
                    k: np.mean(
                        [
                            np.isnan(x.rules.antecedent).sum(axis=1).mean()
                            for x in v
                        ]
                    )
                    for k, v in experiments.items()
                }
                for dataset, experiments in reports.items()
            }

        if measure.capitalize() == "Overlap":
            aux = {
                dataset: {
                    k: np.mean([x.overlap for x in v]) for k, v in experiments.items()
                }
                for dataset, experiments in reports.items()
            }

        if measure.capitalize() == "All":
            experiments = reports[list(reports.keys())[0]]
            aux = {"steps": experiments[list(experiments.keys())[0]]["steps"]}

            aux["F1"] = {
                dataset: {
                    k: np.mean([x.f1 for x in v["report"]])
                    for k, v in experiments.items()
                }
                for dataset, experiments in reports.items()
            }

            aux["Time"] = {
                dataset: {
                    k: np.mean([x.time_calc for x in v["report"]])
                    for k, v in experiments.items()
                }
                for dataset, experiments in reports.items()
            }

            aux["Size"] = {
                dataset: {
                    k: np.mean([len(x.rules_pruned) for x in v["report"]])
                    for k, v in experiments.items()
                }
                for dataset, experiments in reports.items()
            }

            aux["Length"] = {
                dataset: {
                    k: np.mean(
                        [
                            (~np.isnan(x.rules_pruned))
                            .sum(axis=1)
                            .mean()
                            for x in v["report"]
                        ]
                    )
                    for k, v in experiments.items()
                }
                for dataset, experiments in reports.items()
            }

            # aux["Overlap"] = {
            #     dataset: {
            #         k: np.mean([x.overlap for x in v["report"]])
            #         for k, v in experiments.items()
            #     }
            #     for dataset, experiments in reports.items()
            # }

            aux["Overlap"] = {
                dataset: {
                    k: np.mean(
                        [
                            np.isnan(x.rules_pruned)
                            .sum(axis=1)
                            .mean()
                            for x in v["report"]
                        ]
                    )
                    for k, v in experiments.items()
                }
                for dataset, experiments in reports.items()
            }

            return {
                key: pd.DataFrame(values).T.sort_index() for key, values in aux.items()
            }

        return pd.DataFrame(aux).T.sort_index()


class Reports:
    def __init__(self, dataset, setups=None, verbose=False, live=None):
        self.current_report = None
        self.experiment_name = setups.experiment_name
        self.setups = setups
        self.dataset = dataset
        self.reports = dict()
        self.verbose = verbose
        self.live = live
        self.steps_used = set()
        self.order_steps = {'extract': 0, 'interest_measures': 1, 'interest_measures_selection': 2, 'rank': 3, 'prune': 4,
                            'classifier': 5}

    def generate_table(self, current_setup):
        table = Table(header_style="bold magenta")
        table.add_column(f"{self.dataset.capitalize()} ({len(self.setups)})", width=15)
        for step in self.steps_used:
            table.add_column(f"{step.capitalize()}", width=20)

        table.add_column(
            f"Fold: {len(self.reports[current_setup.name]['report'])}/10", width=15
        )
        table.add_column(f"F1", width=15)
        table.add_column(f"Size", width=10)
        table.add_column(f"Time (s)", width=10)

        for report_name, reports in self.reports.items():
            if len(reports) == 0:
                pass

            else:
                if report_name == current_setup.name:
                    row = [f"►"]
                    row.extend(current_setup.get_steps_names())
                    row.extend(
                        [
                            f'{len(reports["report"])}/10',
                            f'{np.mean([r.f1 for r in reports["report"]])}',
                            f'{np.mean([r.size for r in reports["report"]])}',
                            f'{np.mean([r.time for r in reports["report"]])}s',
                        ]
                    )

                    table.add_row(*row)

                else:
                    row = [f" "]
                    row.extend(reports["steps_name"])
                    row.extend(
                        [
                            f'{len(reports["report"])}/10',
                            f'{np.mean([r.f1 for r in reports["report"]])}',
                            f'{np.mean([r.size for r in reports["report"]])}',
                            f'{np.mean([r.time for r in reports["report"]])}s',
                        ]
                    )

                    table.add_row(*row)

        return table

    def last_table(self):
        table = Table(expand=True)
        table.add_column(f"{self.dataset.capitalize()} ", width=5)

        aux = list(self.reports.keys())[-1]

        for step in self.reports[aux]["steps"]:
            table.add_column(f"{step.capitalize()}", width=20)

        table.add_column(f"Finished", width=15)
        table.add_column(f"F1", width=15)
        table.add_column(f"Size", width=10)
        table.add_column(f"Time (s)", width=10)

        for report_name, reports in self.reports.items():
            row = [f" "]

            row.extend(reports["steps_name"])
            row.extend(
                [
                    f'{len(reports["report"])}/10',
                    f'{np.mean([r.f1 for r in reports["report"]])}',
                    f'{np.mean([r.size for r in reports["report"]])}',
                    f'{np.mean([r.time for r in reports["report"]])}s',
                ]
            )

            table.add_row(*row)

        return table

    def refresh_table(self, setup):
        self.live.update(self.generate_table(setup))

    def reset_report(self, report_name):
        if report_name.name in self.reports.keys():
            print(f"Reseting report {report_name.name}")
            self.reports[report_name.name] = []

    def new_report(self, setup):
        report = Report(setup)
        self.steps_used = sorted(set(self.steps_used).union(set(setup.get_steps_used())),
                                 key=lambda x: self.order_steps[x])

        if setup.name not in self.reports.keys():
            self.reports[setup.name] = {
                "report": [report],
                "steps": setup.get_steps_used(),
                "steps_name": setup.get_steps_names(),
            }
        else:
            self.reports[setup.name]["report"].append(report)

        self.current_report = setup

        if self.verbose:
            self.refresh_table(setup)

        report.time_init()
        return report

    def get_data_to_save(self):
        # Return dict of reports with f1
        # return {report.report_name for report in self.reports}
        return self.reports

    def to_csv(self):
        """
        Concat all reports and export to csv
        Returns
        -------

        """
        data = []
        for report_name, folds in self.reports.items():
            for fold in folds["report"]:
                data.append(
                    {
                        "Report": report_name,
                        "F1": fold.f1,
                        "Size": len(fold.rules_pruned),
                        "Time": fold.time,
                    }
                )

        pd.DataFrame(data).groupby(by='Report').mean().to_csv(f"reports/csv/{self.experiment_name}_{self.dataset}.csv")


    def save(self):
        if self.verbose:
            self.live.update(self.last_table())
            self.refresh_table(self.current_report)

        if not os.path.exists(f"reports/raw/{self.experiment_name}"):
            os.makedirs(f"reports/raw/{self.experiment_name}")

        for r_name, folds in self.reports.items():
            for fold in folds["report"]:
                fold.setup = None

        # print(self.get_data_to_save())
        pickle.dump(
            self.get_data_to_save(),
            open(f"reports/raw/{self.experiment_name}/{self.dataset}.pkl", "wb"),
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def load(self):
        self.reports = pickle.load(
            open(f"reports/raw/{self.experiment_name}/{self.dataset}.pkl", "rb")
        )


class Report:
    def __init__(self, setup):
        self.time_calc = 0
        self.start_time = None
        self.setup = setup
        self.report_name = setup.name
        self.time = 0
        self.f1 = 0
        self.size = 0

    def __repr__(self):
        return f"Experiment name: {self.report_name}"

    def save(self, clf, f1, metrics):
        self.rules_pruned = clf.rules_pruned.to_numpy()
        self.support = clf.rules_pruned.support,
        self.confidence = clf.rules_pruned.confidence
        self.f1 = f1
        self.size = len(clf.rules_pruned)
        self.interest_measures = clf.interest_measures if clf.selected_interest_measures is None else clf.selected_interest_measures
        # self.overlap = metrics["overlap"]
        # self.time_calc = metrics["time"]
        # self.size = metrics["size"]
        # self.length = metrics["length"]

        self.time_end()
        del self.start_time

    def time_init(self):
        self.start_time = time.time()

    def time_end(self):
        self.time = time.time()-self.start_time


if __name__ == "__main__":

    experiment_name1 = "ExperimentoFinalFapesp"
    experiment_name2 = "ExperimentoFinalFapespWithoutSelection"
    concat_reports(experiment_name1, experiment_name2)