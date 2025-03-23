from src.preprocessing.dataset import DatasetImporter
from src.processing.config import default_config, bis_config
from src.processing.model import Model
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
from mne import set_log_level
import numpy as np
import random

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


def do_training_all_multicore(importer: DatasetImporter, max_subject):
    max_cpu = cpu_count()
    exp_acc_means = {i: [] for i in range(len(importer.choices))}

    with Pool(processes=max_cpu) as pool:
        tasks = [pool.apply_async(__process_import_exp, (importer, i + 1, max_subject))
            for i in range(len(importer.choices))
        ]
        experiment_data = [task.get() for task in tasks]
        print("All experiments are pre-loaded.")
    
    with ProcessPoolExecutor(max_workers=max_cpu) as pool:
        pre_task = [(exp, subj, data)
            for exp, exp_data in enumerate(experiment_data)
            for subj, data in enumerate(exp_data)
        ]
        random.shuffle((pre_task))
        tasks = [pool.submit(__training, task[0], task[1], task[2])
            for task in pre_task
        ]

        for task in as_completed(tasks):
            result = task.result()
            exp = result[0]
            subj = result[1]
            acc = result[2]
            exp_acc_means[exp].append(acc)

            print(f"experiment {exp:02d}: subject {subj+1:03d}: accuracy {acc:.2f}", flush=True)

    for k, v in exp_acc_means.items():
        print(f"experience {k}: {np.mean(v):.02f}", flush=True)
    print(np.mean(exp_acc_means.values()))

def __training(exp, subj, data):
    set_log_level("CRITICAL")
    model = Model()
    model.load(bis_config())

    result = exp, subj, model.train(data)[0].mean()
    return result

def __process_import_exp(importer, experience, max_subj):
    set_log_level("CRITICAL")
    print(f"Loading experiment {experience}.", flush=True)
    return [importer.get_experience(subj, experience) for subj in range (1, max_subj)]