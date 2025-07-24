from mne.io import Raw
from mne import events_from_annotations, Epochs
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
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
        X, Y = self.__preprocess(raws)

        pipeline = self._grid_cv(X, Y)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=24)
        pipeline.fit(X_train, Y_train)

        self.config.pipeline = pipeline # for being saved
        return pipeline.score(X_test, Y_test)

    @ensure_config
    def cross_validation(self, raws, n_jobs=-1) -> np.ndarray:
        conf = self.config
        X, Y = self.__preprocess(raws)

        pipeline = self._grid_cv(X, Y, n_jobs=n_jobs)

        return cross_val_score(
            pipeline,
            X, Y,
            cv=conf.cross_validator,
            scoring="accuracy",
            n_jobs=n_jobs
        )

    @ensure_config
    def _grid_cv(self, X, Y, n_jobs=-1) -> Pipeline:
        param_grid = {
            'csp__n_components': [2, 4, 6, 8],
        }

        grid = GridSearchCV(
            self.config.pipeline, 
            param_grid=param_grid, 
            cv=self.config.cross_validator, 
            scoring='accuracy', 
            n_jobs=n_jobs)
        grid.fit(X, Y)
        return grid.best_estimator_

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
        events_ids = {"T1": 1, "T2": 2}

        for raw in raws:
            raw.load_data()
            raw = CutFilter().filter(raw, 9, 25)
            events, _ = events_from_annotations(raw, events_ids)
            epochs = Epochs(raw, events, tmin=0.5, tmax=3.5, baseline=None)
            all_X.append(epochs.get_data())
            all_Y.append(epochs.events[:, -1])

        return (np.concatenate(all_X, axis=0), np.concatenate(all_Y, axis=0))
