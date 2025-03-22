from src.processing.model import Model
import numpy as np
import time
import pickle

def do_prediction(raws, model_path):
    model = Model()

    with open(model_path, "rb") as file:
        print(f"Loading model from {model_path}.")
        model.load(pickle.load(file))
        print(f"Model loaded is \n\tpipeline: {model.config.pipeline}\n\tcross_validator: {model.config.cross_validator}")
        print(f"Running prediction.")

        predicted, actual = model.predict(raws)

        print(f"epoch  nb  [prediction]  [truth]  equal?")
        for i, value in enumerate(predicted):
            print(f"epoch [{i:02d}]\t[{value:}]\t  [{actual[i]}]    [{value == actual[i]}]")
            time.sleep(1.5)
        print(f"Accuracy: {np.mean(np.equal(predicted, actual)) * 100 :.2f}%")