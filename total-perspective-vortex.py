from argparse import Namespace
from src.visualize import show_fourrier, show_psd_filter, show_standard, show_filter
from src.train import do_training, do_training_all
from src.predict import do_prediction
from src.preprocessing.dataset import DatasetImporter
from matplotlib import pyplot as pp
from src.tool import define_verbose
import argparse


dataset_arg = """The dataset folder which contains every subjects (or the needed ones). 
The expected structure is dataset/S00X/S00XR0Y.edf with X as the subject number and Y the task number.
If the directory provided is empty a default physionet dataset will be used."""
subject_arg = "The subject to use represented by X."
experiment_arg = """The experiment possibilies are a combination of multiples tasks Y.
T1=[3, 7, 11](open and close left or right fist),
T2=[4, 8, 12](imagine opening and closing left or right fist),
T3=[5, 9, 13](open and close both fists or both feet),
T4=[6, 10, 14](imagine opening and closing both fists or both feet)
T5=[T1, T2]
T6=[T3, T4]"""
experiment_choices = [1, 2, 3, 4, 5, 6]
task_arg = "The task Y to visualize."
task_choices = [i for i in range(3, 15)]
only_arg = "fourier: result of fourier transform, standard: the raw visualization of the dataset, filter: before and after filtering (keeping only 0-30hz freqs)."

functions = {
    "fourier": show_fourrier,
    "standard": show_standard,
    "filter": show_filter,
    "psd_filter": show_psd_filter
}

def train(args: Namespace, importer: DatasetImporter):
    raws = importer.get_experiment(args.subject, args.experiment)

    print(f"Training on subject {args.subject} on experiment {args.experiment} (tasks={importer.choices[args.experiment - 1]})")
    do_training(raws, args.output)

def predict(args: Namespace, importer: DatasetImporter):
    raw = importer.get_task(args.subject, args.task)

    print(f"Predicting on subject {args.subject} on task {args.task}")
    do_prediction([raw], args.model)

def visualize(args: Namespace, importer: DatasetImporter):
    raw = importer.get_task(args.subject, args.task)
    raw.load_data()

    if (getattr(args, "only", False)):
        functions.get(args.only)(raw)
    else:
        for func in functions.values():
            func(raw)
    pp.show()

def all(args: Namespace, importer: DatasetImporter):
    max_subjects=110

    do_training_all(importer, max_subjects)

def main():
    parser = argparse.ArgumentParser("total-perspective-vortex.py", description="EEG signal classification using scikitlearn. This program was developed in the case of a 42 school project.")
    parser.add_argument("--debug", action="store_true", help="Active debugging level")
    parser.add_argument("--dataset", default="dataset", help=dataset_arg)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train", help="Train the model.")
    parser_train.add_argument("subject", help=subject_arg, type=int)
    parser_train.add_argument("experiment", help=experiment_arg, type=int, choices=experiment_choices)
    parser_train.add_argument("--output", default="model.tpv", help="The output file where the model will be stored.")
    parser_train.set_defaults(func=train)

    parser_predict = subparsers.add_parser("predict", help="Use a trained model.")
    parser_predict.add_argument("model", help="The model file to use.")
    parser_predict.add_argument("subject", help=subject_arg, type=int)
    parser_predict.add_argument("task", help=task_arg, type=int, choices=task_choices)
    parser_predict.set_defaults(func=predict)

    parser_visualize = subparsers.add_parser("visualize", help="Vizualise EEG data specific subject task.")
    parser_visualize.add_argument("subject", help=subject_arg, type=int)
    parser_visualize.add_argument("task", help=task_arg, type=int, choices=task_choices)
    parser_visualize.add_argument("--only", choices=list(functions.keys()), help=only_arg)
    parser_visualize.set_defaults(func=visualize)

    parser_all = subparsers.add_parser("all", help="Perfom the mean accuracy test on the whole dataset (may be long).")
    parser_all.set_defaults(func=all)

    result = parser.parse_args()
    define_verbose(result.debug)

    result.func(result, DatasetImporter(result.dataset))

if __name__ == "__main__":
    main()