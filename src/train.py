from src.preprocessing.dataset import DatasetImporter
from src.processing.config import pipeline_ridge
from src.processing.model import Model
import numpy as np

def do_training(raws, output_file):
    model = Model()
    config = pipeline_ridge()

    model.load(config)
    print(f"Loading model with \n\tpipeline: {config.pipeline}\n\tcross_validator: {config.cross_validator}")
    print("Starting model training.")

    score = model.train(raws)
    cross = model.cross_validation(raws)
    print(f"Cross-validation: {cross.round(4)}")
    print(f"Cross-validation accuracy: {cross.mean():.2f}")
    print(f"Model score: {score:.2f}")
    print("Training complete.")

    print(f"Saving model into {output_file}")
    model.save(output_file)


def do_training_all(importer: DatasetImporter, max_subject):
    exp_acc_means = {i: [] for i in range(1, len(importer.choices) + 1)}

    for exp in range(1, len(importer.choices) + 1):
        print(f"loading experiment {exp} all data.")
        exp_data = [importer.get_experiment(subj, exp) for subj in range (1, max_subject)]

        for subj, data in enumerate(exp_data):
            model = Model().load(pipeline_ridge())

            acc = model.train(data)
            exp_acc_means[exp].append(acc)

            print(f"experiment {exp:02d}: subject {subj+1:03d}: accuracy: {acc:.2f}", flush=True)
        
    for k, v in exp_acc_means.items():
        print(f"experiment {k}: {np.mean(v):.02f}", flush=True)
    print(np.mean([np.mean(v) for v in exp_acc_means.values()]))
