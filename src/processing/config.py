
from dataclasses import dataclass
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.pipeline import Pipeline
from src.processing.transformers import CSPTransformer

@dataclass
class Config():
    pipeline: Pipeline
    cross_validator: BaseCrossValidator

def default_config():
    """Generate a default lamda configuration for eeg data."""
    return Config(
        Pipeline([
            ("csp", CSPTransformer(6)),
            ("scaler", StandardScaler()),
            ("gbc", GradientBoostingClassifier(random_state=42))
        ]),
        StratifiedKFold(5, shuffle=True, random_state=42))
