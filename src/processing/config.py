from dataclasses import dataclass
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, KFold, ShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from src.processing.transformers import CSPTransformer
from mne.decoding import CSP

@dataclass
class Config():
    pipeline: Pipeline
    cross_validator: BaseCrossValidator

# marche bien avec LinearSVC
def default_config(seed = 42):
    """Generate a default lamda configuration for eeg data."""
    return Config(
        Pipeline([
            ("csp", CSPTransformer(4)),
            # ("svm", KNeighborsClassifier())
            # ("svm", LinearDiscriminantAnalysis(solver='lsqr'))
            # ("svm", SVC(kernel="linear"))
            # ("lr", LogisticRegression(max_iter=500))
            ("svm", LinearSVC())
            # ("gbc", GradientBoostingClassifier(n_estimators=45, random_state=))
        ]),
        KFold(5, shuffle=True, random_state=seed))

def bis_config():
    """Generate a default lamda configuration for eeg data."""
    return Config(
        Pipeline([
            ("csp", CSP(4, log=True)),
            ("clf", LinearDiscriminantAnalysis()),
        ]),
        ShuffleSplit(10, test_size=0.2, random_state=42))

# def bis_config():
#     """Generate a default lamda configuration for eeg data."""
#     return Config(
#         Pipeline([
#             ("csp", CSPTransformer(4)),
#             ("lda", RandomForestClassifier()),
#         ]),
#         StratifiedKFold(10))