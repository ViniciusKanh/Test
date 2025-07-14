import argparse
from statistics import mean

import marca
import keel_ds as kd
from rich.table import Table
from rich.console import Console
from presets.pipelines import load_pipeline

parser = argparse.ArgumentParser(description="Process datasets and run experiments.")
parser.add_argument('-d', '--dataset', default='balanced', help='Dataset or dataset group')
parser.add_argument('-p', '--pipeline', default='cba,cba_ig,cba_relieff,cba_ensemble',
                    help='Comma separated list of pipelines to run')
parser.add_argument('-v', '--verbose', action='store_true', default=True, help='Verbose output')
parser.add_argument('-w', '--workers', default=1, help='Number of workers for parallel execution')

args = parser.parse_args()

if __name__ == "__main__":
    preset_params = kd.load_preset("preset_for_apriori_10k")["data"]
    datasets = kd.list_data(args.dataset)

    pipelines = [p.strip() for p in args.pipeline.split(",")]
    results = {dataset: {pipe: [] for pipe in pipelines} for dataset in datasets}

    def run_builder(builder, Xtr, ytr, Xte, yte, supp, mlen):
        params = builder.get_params()
        selector = params.pop("interest_measures_selection", None)
        if selector is not None:
            selector.fit(Xtr, ytr)
            Xtr = selector.transform(Xtr)
            Xte = selector.transform(Xte)

        extr = marca.extract.Apriori(
            support=supp, confidence=0, max_len=mlen, remove_redundant=False
        )
        rules = extr(Xtr, ytr)
        clf = marca.ModularClassifier(rules=rules)
        clf.set_params(**params)
        clf.fit(Xtr, ytr)
        return clf.score(Xte, yte)

    for dataset in datasets:
        for data_fold, params in zip(kd.load_data(dataset), preset_params[dataset]):
            x_train, y_train, x_test, y_test = data_fold
            support, maxlen = params["support"], params["maxlen"]

            for pipe in pipelines:
                builders = load_pipeline(pipe).get()
                fold_scores = [
                    run_builder(b, x_train, y_train, x_test, y_test, support, maxlen)
                    for b in builders
                ]
                results[dataset][pipe].append(mean(fold_scores))

    table = Table(title="Accuracy per dataset")
    table.add_column("Dataset")
    for pipe in pipelines:
        table.add_column(pipe)

    for dataset in datasets:
        row = [dataset]
        for pipe in pipelines:
            row.append(f"{mean(results[dataset][pipe])*100:.2f}%")
        table.add_row(*row)

    Console().print(table)
