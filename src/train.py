from src.processing.config import default_config
from src.processing.model import Model

def do_training(raws, output_file):
    model = Model()
    config = default_config()

    model.load(config)
    print(f"Loading model with \n\tpipeline: {config.pipeline}\n\tcross_validator: {config.cross_validator}")
    print("Starting model training.")

    model.train(raws)
    print("Training complete.")
    print(f"Model accuracy {model.evaluate(raws).mean():02f}%", )

    print(f"Saving model into {output_file}")
    model.save(output_file)
