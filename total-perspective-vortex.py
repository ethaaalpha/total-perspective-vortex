import argparse
from argparse import ArgumentParser

dataset_arg = "The dataset folder which contains every subjects (or the needed ones). The expected structure is dataset/S00X/S00XR0Y.edf with X as the subject number and Y the task number."
subject_arg = "The subject to use represented by X."
experience_arg = """The experience possibilies are a combination of multiples tasks Y.
T1=[3, 7, 11](open and close left or right fist),
T2=[4, 8, 12](imagine opening and closing left or right fist),
T3=[5, 9, 13](open and close both fists or both feet),
T4=[6, 10, 14](imagine opening and closing both fists or both feet)"""
experience_choices = [1, 2, 3 ,4]
task_arg = "The task Y to visualize."
only_arg = "fourier: result of fourier transform, standard: the raw visualization of the dataset, filter: before and after filtering."

def train(args):
    print("train" + str(args))

def predict(args):
    print("predict" + str(args))

def visualize(args):
    print("visualize" + str(args))

# [train] [dataset] [subject] [experience] --output-dir=model.json
# [precict] [model] [dataset] [subject] [experience]
# [visualize] [dataset] [subject] [task] --only=[fourier, standard, filter]

def main():
    parser = argparse.ArgumentParser("total-perspective-vortex.py", description="EEG signal classification using scikitlearn. This program was developed in the case of a 42 school project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train", help="Train the model.")
    parser_train.add_argument("dataset", help=dataset_arg)
    parser_train.add_argument("subject", help=subject_arg)
    parser_train.add_argument("experience", help=experience_arg, choices=experience_choices)
    parser_train.add_argument("--output-dir", default="model.json", help="The output file where the dataset will be stored.")
    parser_train.set_defaults(func=train)

    parser_predict = subparsers.add_parser("predict", help="Use a trained model.")
    parser_predict.add_argument("model", help="The model file to use.")
    parser_predict.add_argument("dataset", help=dataset_arg)
    parser_predict.add_argument("subject", help=subject_arg)
    parser_predict.add_argument("experience", help=experience_arg, choices=experience_choices)
    parser_predict.set_defaults(func=predict)

    parser_visualize = subparsers.add_parser("visualize", help="Vizualise EEG data.")
    parser_visualize.add_argument("dataset", help=dataset_arg)
    parser_visualize.add_argument("subject", help=subject_arg)
    parser_visualize.add_argument("task", help=task_arg)
    parser_visualize.add_argument("--only", choices=["fourier", "standard", "filter"], help=only_arg)
    parser_visualize.set_defaults(func=visualize)

    result = parser.parse_args()
    result.func(result)

if __name__ == "__main__":
    main()