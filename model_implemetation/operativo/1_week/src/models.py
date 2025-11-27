"""
Model Definitions and Hyperparameter Grids

Implements SVR, Random Forest, Gradient Boosting, and MLP regressors
with hyperparameter configurations for grid search.
"""

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Dict, List, Any
import numpy as np


class ModelFactory:
    """
    Factory class for creating and configuring prediction models.

    Supports:
    - SVR (Support Vector Regression) with Gaussian kernel
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - MLP (Multi-Layer Perceptron) Regressor
    """

    @staticmethod
    def get_model_configs() -> Dict[str, Dict]:
        """
        Get hyperparameter grids for all models.

        Returns:
            Dictionary with model names as keys and config dicts as values
        """
        configs = {
            'svr_gaussian': {
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01],
                    'epsilon': [0.01, 0.1, 0.5]
                },
                'n_jobs': -1  # Use all cores for SVR
            },

            'random_forest': {
                'param_grid': {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                },
                'n_jobs': -1,
                'random_state': 42
            },

            'gradient_boosting': {
                'param_grid': {
                    'n_estimators': [100, 200, 500],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                'random_state': 42
            },

            'mlp': {
                'param_grid': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01],
                    'max_iter': [500],  # Increased for convergence
                    'early_stopping': [True],
                    'validation_fraction': [0.1]
                },
                'random_state': 42
            }
        }

        return configs

    @staticmethod
    def create_model(model_name: str, params: Dict = None) -> Any:
        """
        Create a model instance with given parameters.

        Args:
            model_name: Name of model ('svr_gaussian', 'random_forest', etc.)
            params: Dictionary of hyperparameters

        Returns:
            Scikit-learn model instance
        """
        if params is None:
            params = {}

        if model_name == 'svr_gaussian':
            return SVR(
                kernel='rbf',  # Gaussian kernel
                C=params.get('C', 1.0),
                gamma=params.get('gamma', 'scale'),
                epsilon=params.get('epsilon', 0.1),
                cache_size=1000  # Increase cache for speed
            )

        elif model_name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                max_features=params.get('max_features', 'sqrt'),
                n_jobs=params.get('n_jobs', -1),
                random_state=params.get('random_state', 42),
                verbose=0
            )

        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                max_depth=params.get('max_depth', 3),
                subsample=params.get('subsample', 1.0),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                random_state=params.get('random_state', 42),
                verbose=0
            )

        elif model_name == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=params.get('hidden_layer_sizes', (100,)),
                activation=params.get('activation', 'relu'),
                alpha=params.get('alpha', 0.0001),
                learning_rate_init=params.get('learning_rate_init', 0.001),
                max_iter=params.get('max_iter', 500),
                early_stopping=params.get('early_stopping', True),
                validation_fraction=params.get('validation_fraction', 0.1),
                random_state=params.get('random_state', 42),
                verbose=False
            )

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    @staticmethod
    def create_multioutput_model(model_name: str, params: Dict = None) -> Any:
        """
        Create a multi-output wrapper for models that don't support it natively.

        For predicting multiple timesteps (24 hours), we need multi-output.
        Random Forest and Gradient Boosting support this natively.
        SVR and MLP need MultiOutputRegressor wrapper.

        Args:
            model_name: Name of model
            params: Dictionary of hyperparameters

        Returns:
            Model capable of multi-output prediction
        """
        base_model = ModelFactory.create_model(model_name, params)

        # Random Forest and Gradient Boosting support multi-output natively
        if model_name in ['random_forest', 'gradient_boosting']:
            return base_model

        # SVR and MLP need wrapper
        else:
            return MultiOutputRegressor(base_model, n_jobs=-1)

    @staticmethod
    def get_all_model_names() -> List[str]:
        """Get list of all available model names."""
        return ['svr_gaussian', 'random_forest', 'gradient_boosting', 'mlp']


def count_total_configurations() -> Dict[str, int]:
    """
    Count total number of hyperparameter combinations for each model.

    Returns:
        Dictionary with model names and configuration counts
    """
    configs = ModelFactory.get_model_configs()
    counts = {}

    for model_name, config in configs.items():
        param_grid = config['param_grid']
        count = 1
        for param_values in param_grid.values():
            count *= len(param_values)
        counts[model_name] = count

    return counts


def estimate_training_time(n_samples: int, n_features: int,
                          n_days: int = 365) -> Dict[str, float]:
    """
    Rough estimate of training time for each model.

    This is very approximate and depends heavily on hardware.

    Args:
        n_samples: Number of training samples per day
        n_features: Number of features
        n_days: Number of days to train (default: 365)

    Returns:
        Dictionary with estimated hours for each model
    """
    # These are rough multipliers based on typical performance
    time_multipliers = {
        'svr_gaussian': 10.0,      # SVR is slowest
        'random_forest': 1.0,      # Baseline
        'gradient_boosting': 2.0,  # Slower than RF
        'mlp': 3.0                 # Depends on architecture
    }

    estimates = {}
    configs_count = count_total_configurations()

    for model_name, multiplier in time_multipliers.items():
        # Rough estimate: seconds per configuration
        base_time = (n_samples * n_features * multiplier) / 1000000

        # Total time = base_time * configs * days
        total_hours = (base_time * configs_count[model_name] * n_days) / 3600

        estimates[model_name] = total_hours

    return estimates


class ModelSelector:
    """
    Manages model selection based on validation performance.

    Keeps track of best models and their parameters across walk-forward iterations.
    """

    def __init__(self):
        self.model_performances = {}
        self.best_model_per_day = {}
        self.best_overall_model = None
        self.best_overall_params = None
        self.best_overall_score = float('inf')  # Lower is better for MSE

    def update(self, date: str, model_name: str, params: Dict,
               score: float, predictions: np.ndarray = None):
        """
        Update performance tracking with new results.

        Args:
            date: Date of prediction
            model_name: Name of model
            params: Hyperparameters used
            score: Performance score (lower is better)
            predictions: Optional predictions array
        """
        # Track all performances
        if model_name not in self.model_performances:
            self.model_performances[model_name] = []

        self.model_performances[model_name].append({
            'date': date,
            'params': params,
            'score': score,
            'predictions': predictions
        })

        # Track best model for this day
        if date not in self.best_model_per_day:
            self.best_model_per_day[date] = {
                'model_name': model_name,
                'params': params,
                'score': score
            }
        elif score < self.best_model_per_day[date]['score']:
            self.best_model_per_day[date] = {
                'model_name': model_name,
                'params': params,
                'score': score
            }

        # Track best overall
        if score < self.best_overall_score:
            self.best_overall_model = model_name
            self.best_overall_params = params
            self.best_overall_score = score

    def get_best_model(self) -> tuple:
        """
        Get best overall model and parameters.

        Returns:
            Tuple of (model_name, params, score)
        """
        return (self.best_overall_model,
                self.best_overall_params,
                self.best_overall_score)

    def get_model_vote_counts(self) -> Dict[str, int]:
        """
        Count how many times each model was best across all days.

        Returns:
            Dictionary with model names and counts
        """
        vote_counts = {}

        for day_info in self.best_model_per_day.values():
            model_name = day_info['model_name']
            vote_counts[model_name] = vote_counts.get(model_name, 0) + 1

        return vote_counts

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for all models.

        Returns:
            Dictionary with statistics for each model
        """
        summary = {}

        for model_name, performances in self.model_performances.items():
            scores = [p['score'] for p in performances]
            summary[model_name] = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'median_score': np.median(scores),
                'n_trials': len(scores)
            }

        return summary


if __name__ == "__main__":
    # Example usage
    print("Model Factory - Configuration Summary")
    print("=" * 60)

    configs = ModelFactory.get_model_configs()
    counts = count_total_configurations()

    for model_name in ModelFactory.get_all_model_names():
        print(f"\n{model_name.upper()}")
        print(f"  Hyperparameter combinations: {counts[model_name]}")
        print(f"  Parameters to tune:")
        for param, values in configs[model_name]['param_grid'].items():
            print(f"    - {param}: {values}")

    print(f"\nTOTAL CONFIGURATIONS: {sum(counts.values())}")

    # Estimate training time
    print("\n" + "=" * 60)
    print("ROUGH TIME ESTIMATES (365 days, 1000 samples, 50 features):")
    estimates = estimate_training_time(1000, 50, 365)
    for model, hours in estimates.items():
        print(f"  {model}: {hours:.1f} hours ({hours/24:.1f} days)")
