from src.processing.model import Model
import pickle

def do_prediction(raws, model_path):
    model = Model()

    with open(model_path, "rb") as file:
        print(f"Loading model from {model_path}.")
        model.load(pickle.load(file))
        print(f"Model loaded is \n\tpipeline: {model.config.pipeline}\n\tcross_validator: {model.config.cross_validator}")

        print(f"Running prediction.")
        model.predict(raws)

