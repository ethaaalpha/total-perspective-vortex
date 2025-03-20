import pickle

def do_prediction(raws, model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
