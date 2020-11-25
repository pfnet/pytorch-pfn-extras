import json
import os
import tempfile
import unittest

import optuna

from pytorch_pfn_extras.config_types import optuna_types
from pytorch_pfn_extras.config_types import load_path_with_optuna_types


class TestConfigTypes(unittest.TestCase):

    study = optuna.create_study()

    def test_config_optuna_types(self):
        def objective(trial):
            types = optuna_types(trial)
            self.assertEqual(
                types['optuna_suggest_categorical'],
                trial.suggest_categorical)
            self.assertEqual(
                types['optuna_suggest_discrete_uniform'],
                trial.suggest_discrete_uniform)
            self.assertEqual(
                types['optuna_suggest_float'],
                trial.suggest_float)
            self.assertEqual(
                types['optuna_suggest_int'],
                trial.suggest_int)
            self.assertEqual(
                types['optuna_suggest_loguniform'],
                trial.suggest_loguniform)
            self.assertEqual(
                types['optuna_suggest_uniform'],
                trial.suggest_uniform)
            return 0.0
        self.study.optimize(objective, n_trials=1)

    def test_load_path_with_optuna_types(self):
        low = 0
        high = 8
        with tempfile.TemporaryDirectory() as temp0:
            with open(os.path.join(temp0, 'foo.json'), mode='w') as f:
                json.dump({
                    'foo': {
                        'type': 'optuna_suggest_int',
                        'name': 'a',
                        'low': low,
                        'high': high
                    }
                }, f)

            def objective(trial):
                config = load_path_with_optuna_types(
                    os.path.join(temp0, 'foo.json'), trial)
                self.assertIsInstance(config['/foo'], int)
                self.assertGreaterEqual(config['/foo'], low)
                self.assertLessEqual(config['/foo'], high)
                return 0.0
            self.study.optimize(objective, n_trials=2 * (high - low + 1))

    def test_load_path_with_optuna_types_with_types_argument(self):
        low = 0
        high = 8
        with tempfile.TemporaryDirectory() as temp0:
            with open(os.path.join(temp0, 'foo.json'), mode='w') as f:
                json.dump({
                    'foo': {
                        'type': 'optuna_suggest_int',
                        'name': 'a',
                        'low': low,
                        'high': high
                    },
                    'bar': {
                        'type': 'dict',
                        'x': 0
                    }
                }, f)

            def objective(trial):
                config = load_path_with_optuna_types(
                    os.path.join(temp0, 'foo.json'), trial,
                    types={'optuna_suggest_int': float, 'dict': dict})
                self.assertIsInstance(config['/foo'], int)
                self.assertGreaterEqual(config['/foo'], low)
                self.assertLessEqual(config['/foo'], high)
                self.assertIsInstance(config['/bar'], dict)
                self.assertEqual(config['/bar']['x'], 0)
                return 0.0
            self.assertWarns(
                UserWarning, self.study.optimize, objective,
                n_trials=2 * (high - low + 1))
