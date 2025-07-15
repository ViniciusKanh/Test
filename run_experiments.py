import argparse
import pandas as pd
import numpy as np
import marca
import keel_ds as kd
from presets.pipelines import load_pipeline


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

    for dataset in datasets:
        index.append(dataset)
        f1_scores = {name: [] for name in pipelines}
        for (x_train, y_train, x_test, y_test), params in zip(
            kd.load_data(dataset), preset_params[dataset]
        ):
            support, maxlen = params["support"], params["maxlen"]
            for name, pipeline in pipelines.items():
                f1 = run_pipeline(
                    pipeline, x_train, y_train, x_test, y_test, support, maxlen
                )
                f1_scores[name].append(f1)
        for name in pipelines:
            results[name].append(np.mean(f1_scores[name]))

    df = pd.DataFrame(results, index=index)
    df.to_excel(args.output)
    print(df)


if __name__ == "__main__":
    main()
