from src.processing.model import Model, DefaultConfig

def do_training(raws, output_file):
    model = Model()
    config = DefaultConfig()

    model.load(config)
    print(f"Loading model with \n\tpipeline: {config.pipeline()}\n\tcross_validator: {config.cross_validator()}")
    print("Starting model training.")

    accuracy = model.train(raws)
    print("Training complete.")
    print(f"Model accuracy {accuracy.mean():02f}%", )

    print(f"Saving model into {output_file}")
    model.save(output_file)
