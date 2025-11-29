# =============================================================================
# File: src/scenarios/scenarios.py
# Description: Centralized scenario definitions for S1 and S2
#
# 📊 SCENARIO DEFINITIONS:
#    Scenario 1 (S1): Predict months 0-23, only pre-LOE features available
#    Scenario 2 (S2): Predict months 6-23, pre-LOE + months 0-5 actuals available
#
# =============================================================================

from typing import Dict, List, Any

# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

SCENARIO_DEFINITIONS: Dict[int, Dict[str, Any]] = {
    1: {
        'name': 'Scenario 1',
        'description': 'Predict months 0-23, only pre-LOE features available',
        'horizon': list(range(0, 24)),  # months 0-23
        'horizon_start': 0,
        'horizon_end': 23,
        'available_months': list(range(-24, 0)),  # months -24 to -1 (pre-LOE)
        'feature_types': ['pre_loe'],
        'early_post_loe_available': False,
        'time_windows': {
            'months_0_5': {'start': 0, 'end': 5, 'metric_weight': 0.5},
            'months_6_11': {'start': 6, 'end': 11, 'metric_weight': 0.2},
            'months_12_23': {'start': 12, 'end': 23, 'metric_weight': 0.1},
        },
        'monthly_metric_weight': 0.2,
    },
    2: {
        'name': 'Scenario 2',
        'description': 'Predict months 6-23, pre-LOE + months 0-5 actuals available',
        'horizon': list(range(6, 24)),  # months 6-23
        'horizon_start': 6,
        'horizon_end': 23,
        'available_months': list(range(-24, 6)),  # months -24 to 5 (pre-LOE + early post)
        'feature_types': ['pre_loe', 'early_post_loe'],
        'early_post_loe_available': True,
        'early_post_loe_months': list(range(0, 6)),  # months 0-5 actuals given
        'time_windows': {
            'months_6_11': {'start': 6, 'end': 11, 'metric_weight': 0.5},
            'months_12_23': {'start': 12, 'end': 23, 'metric_weight': 0.3},
        },
        'monthly_metric_weight': 0.2,
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_scenario_definition(scenario: int) -> Dict[str, Any]:
    """
    Get the full definition for a scenario.
    
    Args:
        scenario: Scenario number (1 or 2)
        
    Returns:
        Dictionary with scenario definition
        
    Raises:
        ValueError: If scenario is not 1 or 2
    """
    if scenario not in SCENARIO_DEFINITIONS:
        raise ValueError(f"Unknown scenario: {scenario}. Must be 1 or 2.")
    return SCENARIO_DEFINITIONS[scenario]


def is_month_in_scenario(month: int, scenario: int) -> bool:
    """
    Check if a month is in the prediction horizon for a scenario.
    
    Args:
        month: Month number (months_postgx)
        scenario: Scenario number (1 or 2)
        
    Returns:
        True if month is in the scenario's prediction horizon
    """
    scenario_def = get_scenario_definition(scenario)
    return month in scenario_def['horizon']


def get_months_for_scenario(scenario: int) -> List[int]:
    """
    Get the list of months to predict for a scenario.
    
    Args:
        scenario: Scenario number (1 or 2)
        
    Returns:
        List of months (months_postgx values)
    """
    return get_scenario_definition(scenario)['horizon']


def get_available_features_for_scenario(scenario: int) -> List[str]:
    """
    Get the types of features available for a scenario.
    
    Args:
        scenario: Scenario number (1 or 2)
        
    Returns:
        List of feature types ('pre_loe' and/or 'early_post_loe')
    """
    return get_scenario_definition(scenario)['feature_types']


def get_scenario_from_month(month: int) -> int:
    """
    Determine which scenario a brand is in based on its earliest available month.
    
    For test data:
    - If data is only up to month -1 → Scenario 1
    - If data is up to month 5 → Scenario 2
    
    Args:
        month: The maximum month available in the data
        
    Returns:
        Scenario number (1 or 2)
    """
    if month < 0:
        return 1
    elif month <= 5:
        return 2
    else:
        # If we have data beyond month 5, it's likely training data
        # Default to S1 (more conservative)
        return 1


def get_time_window_for_month(month: int, scenario: int) -> str:
    """
    Get the time window category for a given month in a scenario.
    
    Args:
        month: Month number (months_postgx)
        scenario: Scenario number (1 or 2)
        
    Returns:
        Time window name (e.g., 'months_0_5', 'months_6_11', 'months_12_23')
        or None if month is not in the scenario's horizon
    """
    scenario_def = get_scenario_definition(scenario)
    
    for window_name, window_def in scenario_def['time_windows'].items():
        if window_def['start'] <= month <= window_def['end']:
            return window_name
    
    return None


def get_scenario_metric_weights(scenario: int) -> Dict[str, float]:
    """
    Get the metric weights for a scenario's time windows.
    
    Args:
        scenario: Scenario number (1 or 2)
        
    Returns:
        Dictionary mapping time window names to metric weights
    """
    scenario_def = get_scenario_definition(scenario)
    weights = {'monthly': scenario_def['monthly_metric_weight']}
    
    for window_name, window_def in scenario_def['time_windows'].items():
        weights[window_name] = window_def['metric_weight']
    
    return weights
