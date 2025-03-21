from dataclasses import dataclass
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from src.processing.transformers import CSPTransformer

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
            ("scaler", StandardScaler()),
            # ("svm", KNeighborsClassifier())
            # ("svm", LinearDiscriminantAnalysis(solver='lsqr'))
            # ("svm", SVC(kernel="linear"))
            ("svm", LinearSVC())
            # ("gbc", GradientBoostingClassifier(n_estimators=45, random_state=))
        ]),
        KFold(5, shuffle=True, random_state=seed))
