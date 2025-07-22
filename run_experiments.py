import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import marca
import keel_ds as kd
from presets.pipelines import load_pipeline
from tqdm import tqdm


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
    return f1


def main():
    parser = argparse.ArgumentParser(description="Run CBA experiments")
    parser.add_argument(
        "--output",
        default="results.xlsx",
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

    results = {name: [] for name in pipelines}
    index = []

    for idx, dataset in enumerate(tqdm(datasets, desc="Datasets"), start=1):
        index.append(f"DS-{idx}")
        f1_scores = {name: [] for name in pipelines}
        for (x_train, y_train, x_test, y_test), params in zip(
            kd.load_data(dataset), preset_params[dataset]
        ):
            support, maxlen = params["support"], params["maxlen"]
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(
                            run_pipeline,
                            pipeline,
                            x_train,
                            y_train,
                            x_test,
                            y_test,
                            support,
                            maxlen,
                        ): name
                        for name, pipeline in pipelines.items()
                    }
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc="Pipelines",
                        leave=False,
                    ):
                        name = futures[future]
                        f1_scores[name].append(future.result())
            else:
                for name, pipeline in tqdm(
                    pipelines.items(), desc="Pipelines", leave=False
                ):
                    f1 = run_pipeline(
                        pipeline,
                        x_train,
                        y_train,
                        x_test,
                        y_test,
                        support,
                        maxlen,
                    )
                    f1_scores[name].append(f1)
        for name in pipelines:
            results[name].append(np.mean(f1_scores[name]))

    df = pd.DataFrame(results, index=index)
    df.to_excel(args.output)
    print(df)


if __name__ == "__main__":
    main()
