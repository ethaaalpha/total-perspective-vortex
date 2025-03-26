from mne.io import Raw
from scipy.stats import randint
from mne import events_from_annotations, Epochs
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate, train_test_split
from src.processing.config import Config
from src.preprocessing.filter import CutFilter
import pickle
import numpy as np

class Model():
    def ensure_config(func):
        def wrapper(cls, *args, **kwargs):
            if cls.config is None:
                raise MemoryError("A model should be loaded before calling this method.")
            return func(cls, *args, **kwargs)
        return wrapper

    @ensure_config
    def train(self, raws) -> float:
        """Return training with on  test data (0.2) score"""
        pipeline = self.config.pipeline
        cv = self.config.cross_validator
        X, Y = self.__preprocess(raws)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=24)

        param_grid = {
            'csp__n_components': [4, 8, 16],
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy")
        grid_search.fit(X_train, Y_train)

        # print(grid_search.best_params_)
        best_pipeline = grid_search.best_estimator_
        # best_pipeline = pipeline
        best_pipeline.fit(X_train, Y_train)

        return best_pipeline.score(X_test, Y_test)

    @ensure_config
    def cross_validation(self, raws) -> np.ndarray:
        conf = self.config
        X, Y = self.__preprocess(raws)

        return cross_val_score(
            conf.pipeline,
            X, Y,
            cv=conf.cross_validator,
            scoring="accuracy"
        )

    @ensure_config
    def predict(self, raws) -> [tuple[np.ndarray, np.ndarray]]:
        """Return predicted and real values"""
        X, Y = self.__preprocess(raws)

        return [self.config.pipeline.predict(X), Y]

    @ensure_config
    def save(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self.config, file)

    def load(self, config: Config):
        self.config = config
        return self

    def __preprocess(self, raws: list[Raw]) -> tuple[np.ndarray, np.ndarray]:
        all_X = []
        all_Y = []
        events_ids = {"T1": 1, "T2": 2} # exclude T0

        for raw in raws:
            raw.load_data()
            raw = CutFilter().filter(raw, 9, 25)
            events, _ = events_from_annotations(raw, events_ids)
            epochs = Epochs(raw, events, tmin=0.5, tmax=3.5, baseline=None)
            all_X.append(epochs.get_data())
            all_Y.append(epochs.events[:, -1])
            # print(epochs.events[:, -1])

        return (np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0))

