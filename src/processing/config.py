from dataclasses import dataclass
from sklearn.calibration import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import ShuffleSplit, BaseCrossValidator, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from src.processing.transformers import CSPCustom
from mne.decoding import CSP

@dataclass
class Config():
    pipeline: Pipeline
    cross_validator: BaseCrossValidator

"""
Notes:
We use StratifiedKFold to ensure good distribution of each classes.
This method is safer against overfitting than ShuffleSplit since folds are not recovered.
"""

def pipeline_tree():
    """Generate CSP + Decision Tree Classifier"""
    return Config(
        Pipeline([
            # ("csp", CSPCustom(4)),  # Include CSP or other feature extractor
            ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
            ("lda", LinearDiscriminantAnalysis())
        ]),
        ShuffleSplit(10, test_size=0.2, random_state=42)
    )

# def pipeline_ridge():
#     """Generate CSP + RidgeClassifier"""
#     return Config(
#         Pipeline([
#             ("ridge", DecisionTree())
#         ]),
#         StratifiedKFold(10, shuffle=True, random_state=24))

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
