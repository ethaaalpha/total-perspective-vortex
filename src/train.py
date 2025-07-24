from joblib import Parallel, delayed
from src.preprocessing.dataset import DatasetImporter
from src.processing.config import pipeline_lda, pipeline_linearsvc, pipeline_ridge
from src.processing.model import Model
from src.tool import define_verbose
import numpy as np


def do_training(raws, output_file):
    model = Model()
    config = pipeline_linearsvc()

    model.load(config)
    print(f"Loading model with \n\tpipeline: {config.pipeline}\n\tcross_validator: {config.cross_validator}")

    score = model.train(raws)
    cross = model.cross_validation(raws)
    print(f"Cross-validation: {cross.round(4)}")
    print(f"Cross-validation score: {cross.mean():.2f}")
    print(f"Model score: {score:.2f}")

    print(f"Saving model into {output_file}")
    model.save(output_file)


def do_training_all(importer: DatasetImporter, max_subject):
    exp_acc_means = {i: [] for i in range(1, len(importer.choices) + 1)}

    def train_subject(subj, exp):
        define_verbose(False)

        data = importer.get_experiment(subj, exp)
        model = Model().load(pipeline_linearsvc())
        acc = model.cross_validation(data, n_jobs=1).mean()
        print(f"experiment {exp:02d}: subject {subj:03d}: accuracy: {acc:.2f}", flush=True)
        return acc

    for exp in range(1, len(importer.choices) + 1):
        print(f"running experiment {exp}")
        accs = Parallel(n_jobs=-1)(delayed(train_subject)(subj, exp) for subj in range(1, max_subject))
        exp_acc_means[exp] = accs

    for k, v in exp_acc_means.items():
        print(f"experiment {k}: {np.mean(v):.02f}", flush=True)
    print(f"Mean accuracy of all experiments: {np.mean([np.mean(v) for v in exp_acc_means.values()]):.3f}")
