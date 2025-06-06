from dataclasses import dataclass
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.pipeline import Pipeline
from src.processing.transformers import CSPCustom

@dataclass
class Config():
    pipeline: Pipeline
    cross_validator: BaseCrossValidator

"""
Notes:
We use StratifiedKFold to ensure good distribution of each classes.
This method is safer against overfitting than ShuffleSplit since folds are not recovered.
"""

def pipeline_ridge():
    """Generate CSP + RidgeClassifier"""
    return Config(
        Pipeline([
            ("csp", CSPCustom()),
            ("ridge", RidgeClassifier())
        ]),
        StratifiedKFold(10, shuffle=True, random_state=24))

def pipeline_linearsvc():
    """Generate CSP + LinearSVC"""
    return Config(
        Pipeline([
            ("csp", CSPCustom()),
            ("svc", LinearSVC())
        ]),
        StratifiedKFold(10, shuffle=True, random_state=24))

def pipeline_lda():
    """Generate CSP + LinearDiscriminantAnalysis"""
    return Config(
        Pipeline([
            ("csp", CSPCustom()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        ]),
        StratifiedKFold(10, shuffle=True, random_state=24))
