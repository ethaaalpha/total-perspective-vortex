from src.processing.model import DefaultModel
import pickle

def do_training(raws, output_file):
    model = DefaultModel(raws)

    model.prepare()
    result = model.fit()
    print(result)
    print(result.mean())

    with open(output_file, "wb") as file:
        pickle.dump(model, file)