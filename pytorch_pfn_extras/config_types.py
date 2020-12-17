import warnings

from .config import Config


def optuna_types(trial):
    types = {
        "optuna_suggest_categorical": trial.suggest_categorical,
        "optuna_suggest_discrete_uniform": trial.suggest_discrete_uniform,
        "optuna_suggest_float": trial.suggest_float,
        "optuna_suggest_int": trial.suggest_int,
        "optuna_suggest_loguniform": trial.suggest_loguniform,
        "optuna_suggest_uniform": trial.suggest_uniform,
    }
    return types


def load_path_with_optuna_types(path, trial, loader=None, types=None):
    if types is None:
        types = {}
    for key, value in optuna_types(trial).items():
        if key in types:
            warnings.warn(key + ' is overwritten by optuna suggest.')
        types[key] = value
    return Config.load_path(path, loader=loader, types=types)
