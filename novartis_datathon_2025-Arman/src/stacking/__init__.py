"""
Bayesian Stacking Module for Novartis Datathon 2025.

Implements Bayesian model averaging / stacking to combine predictions
from multiple base models using the official metric structure.

Main components:
- BayesianStacker: MAP optimization with Dirichlet prior
- HierarchicalBayesianDecay: Hierarchical decay model as extra base
- OOF prediction generation utilities
- Meta-dataset building utilities
- Test prediction ensemble utilities
- Submission diversification utilities
- Official metric evaluation utilities
"""

from .bayesian_stacking import (
    # Core stacker
    BayesianStacker,
    compute_sample_weight,
    compute_sample_weights_vectorized,
    fit_dirichlet_weighted_ensemble,
    
    # OOF and meta-dataset
    generate_oof_predictions,
    build_meta_dataset_for_scenario,
    train_stacker_for_scenario,
    apply_ensemble_to_test,
    build_test_meta_dataset,
    
    # Hierarchical Bayesian Decay (Section 7)
    HierarchicalBayesianDecay,
    generate_oof_for_hierarchical_decay,
    add_hierarchical_decay_to_meta_dataset,
    
    # Diversification (Section 6)
    generate_diversified_submissions,
    fit_stacker_multi_init,
    mcmc_sample_weights,
    generate_bayesian_submission_variants,
    create_blend_of_blends,
    
    # Evaluation (Section 8)
    evaluate_with_official_metric,
    evaluate_oof_predictions,
    compare_models_on_oof,
    evaluate_ensemble_vs_single_models,
)

__all__ = [
    # Core stacker
    "BayesianStacker",
    "compute_sample_weight",
    "compute_sample_weights_vectorized",
    "fit_dirichlet_weighted_ensemble",
    
    # OOF and meta-dataset
    "generate_oof_predictions",
    "build_meta_dataset_for_scenario",
    "train_stacker_for_scenario",
    "apply_ensemble_to_test",
    "build_test_meta_dataset",
    
    # Hierarchical Bayesian Decay
    "HierarchicalBayesianDecay",
    "generate_oof_for_hierarchical_decay",
    "add_hierarchical_decay_to_meta_dataset",
    
    # Diversification
    "generate_diversified_submissions",
    "fit_stacker_multi_init",
    "mcmc_sample_weights",
    "generate_bayesian_submission_variants",
    "create_blend_of_blends",
    
    # Evaluation
    "evaluate_with_official_metric",
    "evaluate_oof_predictions",
    "compare_models_on_oof",
    "evaluate_ensemble_vs_single_models",
]
