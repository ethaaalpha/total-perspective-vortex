from dataclasses import dataclass
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, GridSearchCV, KFold, ShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from src.processing.transformers import CSPTransformer
from mne.decoding import CSP
from mne.preprocessing import ICA

@dataclass
class Config():
    pipeline: Pipeline
    cross_validator: BaseCrossValidator

# marche bien avec LinearSVC
# def default_config(seed = 42):
#     """Generate a default lamda configuration for eeg data."""
#     return Config(
#         Pipeline([
#             ("csp", CSPTransformer(4)),
#             # ("svm", KNeighborsClassifier())
#             # ("svm", LinearDiscriminantAnalysis(solver='lsqr'))
#             # ("svm", SVC(kernel="linear"))
#             # ("lr", LogisticRegression(max_iter=500))
#             ("svm", LinearSVC())
#             # ("gbc", GradientBoostingClassifier(n_estimators=45, random_state=))
#         ]),
#         StratifiedKFold(5, shuffle=True, random_state=seed))

"""
experience 1: 0.61
experience 2: 0.58
experience 3: 0.74
experience 4: 0.63
experience 5: 0.56
experience 6: 0.67
0.6311362759719127
"""
def bis_config():
    """Generate a default lamda configuration for eeg data."""
    return Config(
        Pipeline([
            ("csp", CSP(log=True)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
        ShuffleSplit(10, test_size=0.2, random_state=24))

# # 0.61
# def bis_config():
#     """Generate a default lamda configuration for eeg data."""
#     return Config(
#         Pipeline([
#             ("csp", CSP(6, log=True)),
#             ("clf", LinearDiscriminantAnalysis()),
#         ]),
#         ShuffleSplit(10, test_size=0.2, random_state=42))

# def bis_config():
#     """Generate a default lamda configuration for eeg data."""
#     return Config(
#         Pipeline([
#             ("csp", CSPTransformer(4)),
#             ("lda", RandomForestClassifier()),
#         ]),
#         StratifiedKFold(10))