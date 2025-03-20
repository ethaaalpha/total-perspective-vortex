import pickle

from processing.model import AbstractModel

def do_prediction(raws, model_path):
    with open(model_path, "rb") as file:
        model: AbstractModel = pickle.load(file)
        print(model.)