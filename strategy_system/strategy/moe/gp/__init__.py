"""Bounded genetic programming for causal, state-conditioned features."""

from .evolution import (
    GPEvolutionResult,
    GeneticFeatureModel,
    SelectedExpression,
    build_gp_inputs,
    extrapolate_features,
)
from .grammar import Node, evaluate_expression, expression_partial
from .grammar import (calculate_expression_complexity, calculate_expression_depth,
                      get_operator_rules, get_operators, get_primitive_features,
                      validate_expression)
from .population import (create_initial_population, create_random_expression,
                         deduplicate_population, filter_constant_features,
                         filter_invalid_features, filter_redundant_features)
from .evolution import (evaluate_population, select_parents, crossover, mutate,
                        evolve_generation, evolve_population, stop_evolution)
from .evaluation import (prefilter_features, test_feature, test_feature_set,
                         run_walk_forward_test, run_purged_walk_forward,
                         calculate_feature_stability, calculate_model_stability,
                         calculate_prediction_error, calculate_directional_accuracy,
                         calculate_expectancy, calculate_complexity_penalty,
                         calculate_fitness, fit_predictive_model, ridge_select_features,
                         canonical_model_name, validate_evaluation_models)
from .feature_library import (build_feature_library, select_validated_features,
                              save_feature_library, load_feature_library,
                              get_features_for_regime, register_regime_features)

__all__ = [
    "GPEvolutionResult",
    "GeneticFeatureModel",
    "SelectedExpression",
    "build_gp_inputs",
    "extrapolate_features",
    "Node",
    "evaluate_expression",
    "expression_partial",
    "get_primitive_features", "get_operators", "get_operator_rules", "validate_expression",
    "calculate_expression_depth", "calculate_expression_complexity",
    "create_random_expression", "create_initial_population", "deduplicate_population",
    "filter_invalid_features", "filter_constant_features", "filter_redundant_features",
    "evaluate_population", "select_parents", "crossover", "mutate", "evolve_generation",
    "evolve_population", "stop_evolution", "prefilter_features", "test_feature",
    "test_feature_set", "run_walk_forward_test", "run_purged_walk_forward",
    "calculate_feature_stability", "calculate_model_stability", "calculate_prediction_error",
    "calculate_directional_accuracy", "calculate_expectancy", "calculate_complexity_penalty",
    "calculate_fitness", "fit_predictive_model", "ridge_select_features",
    "canonical_model_name", "validate_evaluation_models",
    "build_feature_library", "select_validated_features", "save_feature_library",
    "load_feature_library", "get_features_for_regime", "register_regime_features",
]
