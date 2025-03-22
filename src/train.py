from src.processing.config import default_config
from src.processing.model import Model
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

def do_training_all(subjects, experiment):
    model = Model()
    acc_historic = []
    config = default_config()
    
    model.load(config)
    for i, subject in enumerate(subjects):
        accuracy, _ = model.train(subject)
        acc_historic.append(accuracy.mean())
        print(f"experiment {experiment:02d}: subject {i + 1:03d}: accuracy {accuracy.mean():.2f}")
    return np.mean(acc_historic)