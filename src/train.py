from src.processing.model import DefaultModel

def do_training(raws):
    model = DefaultModel(raws)

    model.prepare()
    result = model.fit()
    print(result)
    print(result.mean())
