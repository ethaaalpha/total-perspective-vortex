from src.preprocessing.dataset import DatasetImporter
from src.processing.config import bis_config, default_config
from src.processing.model import Model
from concurrent.futures import as_completed, ProcessPoolExecutor
from mne import set_log_level
import os
import numpy as np

def do_training(raws, output_file):
    model = Model()
    config = default_config()

    model.load(config)
    print(f"Loading model with \n\tpipeline: {config.pipeline}\n\tcross_validator: {config.cross_validator}")
    print("Starting model training.")

    accuracy, score = model.train(raws)
    print(f"Cross-validation accuracy: {accuracy.mean():.2f}", )
    print(f"Model score: {score:.2f}")
    print("Training complete.")

    print(f"Saving model into {output_file}")
    model.save(output_file)


def do_training_all(importer: DatasetImporter, max_subject):
    exp_acc_means = {i: [] for i in range(1, len(importer.choices) + 1)}

    for exp in range(1, len(importer.choices) + 1):
        print(f"loading experiment {exp} all data.")
        exp_data = [importer.get_experience(subj, exp) for subj in range (1, max_subject)]

        all_args = [(exp, subj, data)
            for subj, data in enumerate(exp_data)
        ]

        for task_args in all_args:
            result = __training(*task_args)
            exp = result[0]
            subj = result[1]
            acc = result[2]
            exp_acc_means[exp].append(acc)

            print(f"experiment {exp:02d}: subject {subj+1:03d}: accuracy {acc:.2f}", flush=True)
        
        print(f"experience {exp}: {np.mean(exp_acc_means.get(exp))}")

    for k, v in exp_acc_means.items():
        print(f"experience {k}: {np.mean(v):.02f}", flush=True)
    print(np.mean(exp_acc_means.values()))

def __training(exp, subj, data):
    model = Model()
    model.load(bis_config())

    data = model.train(data, False)[0]
    result = exp, subj, data.mean() # we want accuracy and not cross_val_score

    # Print scores for each fold

    from matplotlib import pyplot as plt
    # Plot the scores
    plt.plot(range(1, len(data) + 1), data, marker='o')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Cross-validation Score Over Folds')
    plt.show()

    return result

def __process_import_exp(importer, experience, max_subj):
    set_log_level("CRITICAL")
    print(f"Loading experiment {experience}.", flush=True)
    return 