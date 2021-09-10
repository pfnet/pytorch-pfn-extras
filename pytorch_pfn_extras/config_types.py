import warnings
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from pytorch_pfn_extras import config


if TYPE_CHECKING:
    import optuna


def optuna_types(trial: 'optuna.trial.Trial') -> Dict[str, Any]:
    types = {
        "optuna_suggest_categorical": trial.suggest_categorical,
        "optuna_suggest_discrete_uniform": trial.suggest_discrete_uniform,
        "optuna_suggest_float": trial.suggest_float,
        "optuna_suggest_int": trial.suggest_int,
        "optuna_suggest_loguniform": trial.suggest_loguniform,
        "optuna_suggest_uniform": trial.suggest_uniform,
    }
    return types


def load_path_with_optuna_types(
        path: str,
        trial: 'optuna.trial.Trial',
        loader: Optional[config.Loader] = None,
        types: Optional[Dict[str, Callable[..., Any]]] = None,
) -> config.Config:
    if types is None:
        types = {}
    for key, value in optuna_types(trial).items():
        if key in types:
            warnings.warn(key + ' is overwritten by optuna suggest.')
        types[key] = value
    return config.Config.load_path(path, loader=loader, types=types)
