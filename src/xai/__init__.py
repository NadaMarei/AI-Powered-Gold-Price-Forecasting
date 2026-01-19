# XAI Module
from .shap_explainer import (
    SHAPExplainer,
    compute_temporal_importance,
    aggregate_feature_importance
)
from .stability_analysis import (
    ExplanationStabilityAnalyzer,
    compute_stability_metrics
)
from .counterfactual import (
    CounterfactualGenerator,
    generate_economic_scenarios
)

__all__ = [
    'SHAPExplainer',
    'compute_temporal_importance',
    'aggregate_feature_importance',
    'ExplanationStabilityAnalyzer',
    'compute_stability_metrics',
    'CounterfactualGenerator',
    'generate_economic_scenarios'
]
