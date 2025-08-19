import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import marca
import keel_ds as kd
from presets.pipelines import load_pipeline
from tqdm import tqdm


# NOTE: This script mirrors run_experiments.py but additionally records
# the size of the generated model (number of rules) using `get_size`.

def run_pipeline(pipeline, x_train, y_train, x_test, y_test, support, maxlen):
    params = pipeline.get_params()
    selector = params.pop("interest_measures_selection", None)
    if selector is not None:
        selector.fit(x_train, y_train)
        x_train_fs = selector.transform(x_train)
        x_test_fs = selector.transform(x_test)
    else:
        x_train_fs, x_test_fs = x_train, x_test

    extr = marca.extract.Apriori(
        support=support,
        confidence=0,
        max_len=maxlen,
        remove_redundant=False,
    )
    rules = extr(x_train_fs, y_train)
    clf = marca.ModularClassifier(rules=rules)
    clf.set_params(**params)
    clf.fit(x_train_fs, y_train)
    f1 = clf.score(x_test_fs, y_test)
    #-------Tamanho do Modelo-----
    size = clf.get_size()
    return f1, size


def process_dataset(idx, dataset, params_dataset, pipelines):
    """Run all pipelines for all folds of a dataset."""
    f1_scores = {name: [] for name in pipelines}
    model_sizes = {name: [] for name in pipelines}
    for (x_train, y_train, x_test, y_test), params in zip(
        kd.load_data(dataset), params_dataset
    ):
        support, maxlen = params["support"], params["maxlen"]
        for name, pipeline in pipelines.items():
            f1, size = run_pipeline(
                pipeline,
                x_train,
                y_train,
                x_test,
                y_test,
                support,
                maxlen,
            )
            f1_scores[name].append(f1)
            model_sizes[name].append(size)

    means = {
        name: {"f1": np.mean(f1_scores[name]), "size": np.mean(model_sizes[name])}
        for name in pipelines
    }
    return idx, means


def main():
    parser = argparse.ArgumentParser(
        description="Run CBA experiments including model size"
    )
    parser.add_argument(
        "--output",
        default="results_with_size.xlsx",
        help="Output Excel file with results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of processes to use for pipeline execution",
    )
    args = parser.parse_args()

    preset_params = kd.load_preset("preset_for_apriori_10k")["data"]
    datasets = kd.list_data("balanced")

    pipelines = {
        "CBA": load_pipeline("cba").get()[0],
        "CBA_IG": load_pipeline("cba_ig").get()[0],
        "CBA_ReliefF": load_pipeline("cba_relieff").get()[0],
        "CBA_Ensemble": load_pipeline("cba_ensemble").get()[0],
    }

    results_tmp = {name: {"f1": {}, "size": {}} for name in pipelines}

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_dataset,
                    idx,
                    dataset,
                    preset_params[dataset],
                    pipelines,
                ): idx
                for idx, dataset in enumerate(datasets, start=1)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Datasets",
            ):
                idx, row = future.result()
                for name in pipelines:
                    results_tmp[name]["f1"][idx] = row[name]["f1"]
                    results_tmp[name]["size"][idx] = row[name]["size"]
    else:
        for idx, dataset in enumerate(
            tqdm(datasets, desc="Datasets"), start=1
        ):
            _, row = process_dataset(
                idx, dataset, preset_params[dataset], pipelines
            )
            for name in pipelines:
                results_tmp[name]["f1"][idx] = row[name]["f1"]
                results_tmp[name]["size"][idx] = row[name]["size"]

    index = [f"DS-{i}" for i in sorted(results_tmp[next(iter(results_tmp))]["f1"])]
    results = {}
    for name in pipelines:
        results[f"{name}_F1"] = [
            results_tmp[name]["f1"][i] for i in sorted(results_tmp[name]["f1"])
        ]
        results[f"{name}_Size"] = [
            results_tmp[name]["size"][i] for i in sorted(results_tmp[name]["size"])
        ]

    df = pd.DataFrame(results, index=index)
    df.to_excel(args.output)
    print(df)


if __name__ == "__main__":
    main()
