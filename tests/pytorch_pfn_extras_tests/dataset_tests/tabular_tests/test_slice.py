import itertools
import warnings

import numpy as np
import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras_tests.dataset_tests.tabular_tests import dummy_dataset  # NOQA


def _values_to_dicts(names, values):
    assert isinstance(names, str)
    assert isinstance(values, (tuple, list))

    def safe_zip(ns, vs):
        if len(ns) == 1:
            return [(ns[0], vs)]
        assert isinstance(vs, (tuple, list)) and len(ns) == len(vs)
        return zip(ns, vs)

    names = names.split(',')
    params = [dict(safe_zip(names, value_list)) for value_list in values]
    return params


def product(parameter):
    if isinstance(parameter, dict):
        return product_dict(*[
            _values_to_dicts(names, values)
            for names, values in sorted(parameter.items())])

    elif isinstance(parameter, list):
        # list of lists of dicts
        if not all(isinstance(_, list) for _ in parameter):
            raise TypeError('parameter must be list of lists of dicts')
        if not all(isinstance(_, dict) for l in parameter for _ in l):  # NOQA
            raise TypeError('parameter must be list of lists of dicts')
        return product_dict(*parameter)

    else:
        raise TypeError(
            'parameter must be either dict or list. Actual: {}'.format(
                type(parameter)))


def product_dict(*parameters):
    return [
        {k: v for dic in dicts for k, v in dic.items()}
        for dicts in itertools.product(*parameters)]


def _filter_params(params):
    for param in params:
        if 'expected_len' in param and \
           isinstance(param['get_examples_indices'], list) and \
           any(param['expected_len'] <= index
               for index in param['get_examples_indices']):
            continue

        if 'expected_keys' in param and \
           isinstance(param['get_examples_key_indices'], tuple) and \
           any(len(param['expected_keys']) <= key_index
               for key_index in param['get_examples_key_indices']):
            continue

        # To reduce the number of tests,
        # drop combinations of indices and keys.
        # (check only `slice[indices]` and `slice[:, keys]`)
        if (not (param['indices'] == slice(None)
                 and param['get_examples_indices'] is None)
                and not (param['keys'] is None
                         and param['get_examples_key_indices'] is None)):
            continue

        yield param


params = _filter_params(product_dict(
    product_dict(
        [{'mode': tuple}, {'mode': dict}],
        [
            {'keys': None, 'expected_keys': ('a', 'b', 'c')},
            {'keys': 1, 'expected_keys': ('b',)},
            {'keys': (1,), 'expected_keys': ('b',)},
            {'keys': 3, 'key_exception': IndexError},
            {'keys': (3,), 'key_exception': IndexError},
            {'keys': 'c', 'expected_keys': ('c',)},
            {'keys': ('c',), 'expected_keys': ('c',)},
            {'keys': 'd', 'key_exception': KeyError},
            {'keys': ('d',), 'key_exception': KeyError},
            {'keys': (-1, 'a'), 'expected_keys': ('c', 'a')},
            {'keys': (), 'expected_keys': ()},
        ],
    )
    + product_dict(
        [{'mode': None}],
        [
            {'keys': None, 'expected_keys': ('a',)},
            {'keys': 0, 'expected_keys': ('a',)},
            {'keys': (0,), 'expected_keys': ('a',)},
            {'keys': 1, 'key_exception': IndexError},
            {'keys': (1,), 'key_exception': IndexError},
            {'keys': 'a', 'expected_keys': ('a',)},
            {'keys': ('a',), 'expected_keys': ('a',)},
            {'keys': 'b', 'key_exception': KeyError},
            {'keys': ('b',), 'key_exception': KeyError},
            {'keys': (), 'expected_keys': ()},
        ],
    ),
    product({
        'return_array': [True, False],
        'integer': [int, np.int32],
    }),
    [
        {'indices': slice(None), 'expected_len': 10},
        {'indices': [3, -2], 'expected_len': 2},
        {'indices': [11, 1], 'index_exception': IndexError},
        {'indices': [i in {1, 3} for i in range(10)], 'expected_len': 2},
        {'indices': [True] * 11, 'index_exception': ValueError},
        {'indices': slice(3, None, -2), 'expected_len': 2},
        {'indices': [False, 3, 9, 5, True], 'expected_len': 5},
        {'indices': [], 'expected_len': 0},
    ],
    product({
        'get_examples_indices': [
            None, [1], [1, 0], slice(0, 2, 1), slice(1, None, -1), []],
        'get_examples_key_indices': [None, (1,), (1, 0), ()],
    }),
))


@pytest.mark.parametrize(
    'test_args',
    params
)
def test_slice(test_args):
    exception = test_args.get('index_exception', None) \
        or test_args.get('key_exception', None)

    indices = test_args['indices']
    keys = test_args['keys']
    mode = test_args['mode']
    return_array = test_args['return_array']
    get_examples_indices = test_args['get_examples_indices']
    get_examples_key_indices = test_args['get_examples_key_indices']

    if isinstance(indices, list):
        indices = [
            index if isinstance(index, bool) else test_args['integer'](index)
            for index in indices]

    def callback(indices, key_indices):
        if isinstance(indices, list) \
                or isinstance(get_examples_indices, list):
            assert isinstance(indices, list)
        elif isinstance(indices, slice) \
                or isinstance(get_examples_indices, slice):
            assert isinstance(indices, slice)
        else:
            assert indices is None

        if keys is None and get_examples_key_indices is None:
            assert key_indices is None
        else:
            assert isinstance(key_indices, tuple)

    dataset = dummy_dataset.DummyDataset(
        mode=mode, return_array=return_array, callback=callback,
        convert=True)

    if exception is not None:
        with pytest.raises(exception):
            if keys is None:
                dataset.slice[indices]
            else:
                dataset.slice[indices, keys]
        return

    if keys is None:
        view = dataset.slice[indices]
        data = dataset.data[:, _indices_for_numpy(indices)]
    else:
        view = dataset.slice[indices, keys]
        if isinstance(keys, tuple):
            keys = keys
        else:
            keys = keys,
        key_indices = [
            {'a': 0, 'b': 1, 'c': 2}.get(key, key) for key in keys]
        data = dataset.data[key_indices][
            :, _indices_for_numpy(indices)]

    assert isinstance(view, ppe.dataset.TabularDataset)
    assert len(view) == test_args['expected_len']
    assert view.keys == test_args['expected_keys']
    if keys is None:
        assert view.mode == mode
    elif isinstance(keys, tuple):
        assert view.mode == mode or tuple
    else:
        assert view.mode is None

    output = view.get_examples(
        get_examples_indices, get_examples_key_indices)

    if get_examples_indices is not None:
        data = data[:, _indices_for_numpy(get_examples_indices)]
    if get_examples_key_indices is not None:
        data = data[list(get_examples_key_indices)]

    for out, d in itertools.zip_longest(output, data):
        np.testing.assert_equal(out, d)
        if return_array:
            assert isinstance(out, np.ndarray)
        else:
            assert isinstance(out, list)

    assert view.convert(output) == 'converted'


# Replace list of bool with ndarray of bool
# since old numpy cannot handle list of bool.
def _indices_for_numpy(indices):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        if len(np.empty(2)[[False, True]]) == 1:
            # new numpy
            return indices

    # old numpy
    if isinstance(indices, list) and \
       len(indices) > 0 and \
       isinstance(indices[0], bool):
        return np.array(indices)
    else:
        return indices
