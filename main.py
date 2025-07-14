import marca
import argparse
import keel_ds as kd
from presets.pipelines import load_pipeline

parser = argparse.ArgumentParser(description="Process datasets and run experiments.")
parser.add_argument('-d', '--dataset', help='Run experiment on dataset or set of datasets')
parser.add_argument('-p', '--pipeline', default='default', help='Setup for experiments')
parser.add_argument('-v', '--verbose', action='store_true', default=True, help='Verbose output')
parser.add_argument('-w', '--workers', default=1, help='Number of workers for parallel execution')

args = parser.parse_args()

if __name__ == "__main__":
    preset_params = kd.load_preset("preset_for_apriori_10k")['data']
    datasets = kd.list_data("balanced")

    for dataset in datasets:
        for data_fold, params in zip(kd.load_data(dataset), preset_params[dataset]):
            x_train, y_train, x_test, y_test = data_fold
            support, maxlen = params["support"], params["maxlen"]

            for pipeline in load_pipeline(args.pipeline).get():
                params = pipeline.get_params()
                selector = params.pop("interest_measures_selection", None)
                if selector is not None:
                    selector.fit(x_train, y_train)
                    x_train_fs = selector.transform(x_train)
                    x_test_fs = selector.transform(x_test)
                else:
                    x_train_fs, x_test_fs = x_train, x_test

                extr = marca.extract.Apriori(support=support, confidence=0, max_len=maxlen, remove_redundant=False)
                rules = extr(x_train_fs, y_train)
                clf = marca.ModularClassifier(rules=rules)
                clf.set_params(**params)
                clf.fit(x_train_fs, y_train)
                print(clf.score(x_test_fs, y_test))
