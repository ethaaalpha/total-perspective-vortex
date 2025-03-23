from argparse import Namespace
from src.visualize import show_fourrier, show_standard, show_filter
from src.train import do_training, do_training_all_multicore
from src.predict import do_prediction
from src.preprocessing.dataset import DatasetImporter
from matplotlib import pyplot as pp
import argparse
import mne
import numpy as np

dataset_arg = """The dataset folder which contains every subjects (or the needed ones). 
The expected structure is dataset/S00X/S00XR0Y.edf with X as the subject number and Y the task number.
If the directory provided is empty a default physionet dataset will be used."""
subject_arg = "The subject to use represented by X."
experience_arg = """The experience possibilies are a combination of multiples tasks Y.
T1=[3, 7, 11](open and close left or right fist),
T2=[4, 8, 12](imagine opening and closing left or right fist),
T3=[5, 9, 13](open and close both fists or both feet),
T4=[6, 10, 14](imagine opening and closing both fists or both feet)
T5=[T1, T2]
T6=[T3, T4]"""
experience_choices = [1, 2, 3, 4, 5, 6]
task_arg = "The task Y to visualize."
task_choices = [i for i in range(1, 15)]
only_arg = "fourier: result of fourier transform, standard: the raw visualization of the dataset, filter: before and after filtering (keeping only 0-30hz freqs)."

def train(args: Namespace):
    raws = DatasetImporter(args.dataset).get_experience(args.subject, args.experience)

    do_training(raws, args.output)

def predict(args: Namespace):
    raw = DatasetImporter(args.dataset).get_task(args.subject, args.task)

    do_prediction([raw], args.model)

def visualize(args: Namespace):
    raw = DatasetImporter(args.dataset).get_task(args.subject, args.task)

    functions = {
        "fourier": show_fourrier,
        "standard": show_standard,
        "filter": show_filter
    }

    if (getattr(args, "only", False)):
        functions.get(args.only)(raw)
    else:
        for func in functions.values():
            func(raw)
    pp.show()

def all(args: Namespace):
    importer = DatasetImporter(args.dataset)
    max_subjects=110

    do_training_all_multicore(importer, max_subjects)
        
def define_verbose(debug: bool):
    if not debug:
        mne.set_log_level("CRITICAL")

# [train] [dataset] [subject] [experience] --output-dir=model.json
# [precict] [model] [dataset] [subject] [experience]
# [visualize] [dataset] [subject] [task] --only=[fourier, standard, filter]
# [all] [dataset]
# 225 sec

def main():
    parser = argparse.ArgumentParser("total-perspective-vortex.py", description="EEG signal classification using scikitlearn. This program was developed in the case of a 42 school project.")
    parser.add_argument("--debug", action="store_true", help="Active debugging level")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train", help="Train the model.")
    parser_train.add_argument("dataset", help=dataset_arg)
    parser_train.add_argument("subject", help=subject_arg, type=int)
    parser_train.add_argument("experience", help=experience_arg, type=int, choices=experience_choices)
    parser_train.add_argument("--output", default="model.json", help="The output file where the dataset will be stored.")
    parser_train.set_defaults(func=train)

    parser_predict = subparsers.add_parser("predict", help="Use a trained model.")
    parser_predict.add_argument("model", help="The model file to use.")
    parser_predict.add_argument("dataset", help=dataset_arg)
    parser_predict.add_argument("subject", help=subject_arg, type=int)
    parser_predict.add_argument("task", help=task_arg, type=int, choices=task_choices)
    parser_predict.set_defaults(func=predict)

    parser_visualize = subparsers.add_parser("visualize", help="Vizualise EEG data specific subject task.")
    parser_visualize.add_argument("dataset", help=dataset_arg)
    parser_visualize.add_argument("subject", help=subject_arg, type=int)
    parser_visualize.add_argument("task", help=task_arg, type=int, choices=task_choices)
    parser_visualize.add_argument("--only", choices=["fourier", "standard", "filter"], help=only_arg)
    parser_visualize.set_defaults(func=visualize)

    parser_all = subparsers.add_parser("all", help="Perfom the mean accuracy test on the whole dataset (may be long).")
    parser_all.add_argument("dataset", help=dataset_arg)
    parser_all.set_defaults(func=all)

    result = parser.parse_args()
    define_verbose(result.debug)

    result.func(result)

if __name__ == "__main__":
    main()