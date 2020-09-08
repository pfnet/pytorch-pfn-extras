import numpy as np
import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.dataset import tabular


class TestFromData:

    def test_unary_array(self):
        dataset = tabular.from_data(np.arange(10))

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert len(dataset.keys) == 1
        assert dataset is not None

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [1, 3])
        assert isinstance(output, np.ndarray)

    def test_unary_array_with_key(self):
        dataset = tabular.from_data(('a', np.arange(10)))

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert dataset.keys == ('a',)
        assert dataset.mode is None

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [1, 3])
        assert isinstance(output, np.ndarray)

    def test_unary_list(self):
        dataset = tabular.from_data([2, 7, 1, 8, 4, 5, 9, 0, 3, 6])

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert len(dataset.keys) == 1
        assert dataset.mode is None

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [7, 8])
        assert isinstance(output, list)

    def test_unary_list_with_key(self):
        dataset = tabular.from_data(('a', [2, 7, 1, 8, 4, 5, 9, 0, 3, 6]))

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert dataset.keys == ('a',)
        assert dataset.mode is None

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [7, 8])
        assert isinstance(output, list)

    def test_unary_callable_unary(self):
        dataset = tabular.from_data(('a', lambda i: i * i), size=10)

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert dataset.keys == ('a',)
        assert(dataset.mode) is None

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [1, 9])
        assert isinstance(output, list)

    def test_unary_callable_tuple(self):
        dataset = tabular.from_data(
            (('a', 'b'), lambda i: (i * i, -i)), size=10)

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert dataset.keys == ('a', 'b')
        assert dataset.mode == tuple

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 9], [-1, -3]))
        for out in output:
            assert isinstance(out, list)

    def test_unary_callable_dict(self):
        dataset = tabular.from_data(
            (('a', 'b'), lambda i: {'a': i * i, 'b': -i}), size=10)

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert dataset.keys == ('a', 'b')
        assert dataset.mode == dict

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, {'a': [1, 9], 'b': [-1, -3]})
        for out in output.values():
            assert isinstance(out, list)

    def test_unary_callable_without_key(self):
        with pytest.raises(ValueError):
            tabular.from_data(lambda i: i * i, size=10)

    def test_unary_callable_without_size(self):
        with pytest.raises(ValueError):
            tabular.from_data(('a', lambda i: i * i))

    def test_tuple_array_list(self):
        dataset = tabular.from_data(
            (np.arange(10), [2, 7, 1, 8, 4, 5, 9, 0, 3, 6]))

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert len(dataset.keys) == 2
        assert dataset.mode == tuple

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [7, 8]))
        assert isinstance(output[0], np.ndarray)
        assert isinstance(output[1], list)

    def test_tuple_array_with_key_list(self):
        dataset = tabular.from_data(
            (('a', np.arange(10)), [2, 7, 1, 8, 4, 5, 9, 0, 3, 6]))

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert len(dataset.keys) == 2
        assert dataset.keys[0] == 'a'
        assert dataset.mode == tuple

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [7, 8]))
        assert isinstance(output[0], np.ndarray)
        assert isinstance(output[1], list)

    def test_tuple_array_list_with_key(self):
        dataset = tabular.from_data(
            (np.arange(10), ('b', [2, 7, 1, 8, 4, 5, 9, 0, 3, 6])))

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert len(dataset.keys) == 2
        assert dataset.keys[1] == 'b'
        assert dataset.mode == tuple

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [7, 8]))
        assert isinstance(output[0], np.ndarray)
        assert isinstance(output[1], list)

    def test_tuple_array_callable_unary(self):
        dataset = tabular.from_data((np.arange(10), ('b', lambda i: i * i)))

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert len(dataset.keys) == 2
        assert dataset.keys[1] == 'b'
        assert dataset.mode == tuple

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 9]))
        assert isinstance(output[0], np.ndarray)
        assert isinstance(output[1], list)

    def test_tuple_array_callable_tuple(self):
        dataset = tabular.from_data(
            (np.arange(10), (('b', 'c'), lambda i: (i * i, -i))))

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert len(dataset.keys) == 3
        assert dataset.keys[1] == ('b')
        assert dataset.mode == tuple

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 9], [-1, -3]))
        assert isinstance(output[0], np.ndarray)
        assert isinstance(output[1], list)

    def test_tuple_array_callable_dict(self):
        dataset = tabular.from_data(
            (np.arange(10), (('b', 'c'), lambda i: {'b': i * i, 'c': -i})))

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert len(dataset.keys) == 3
        assert dataset.keys[1] == ('b')
        assert dataset.mode == tuple

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 9], [-1, -3]))
        assert isinstance(output[0], np.ndarray)
        assert isinstance(output[1], list)

    def test_tuple_array_with_key_callable_unary(self):
        dataset = tabular.from_data(
            (('a', np.arange(10)), ('b', lambda i: i * i)))

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert dataset.keys == ('a', 'b')
        assert dataset.mode == tuple

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 9]))
        assert isinstance(output[0], np.ndarray)
        assert isinstance(output[1], list)

    def test_tuple_callable_unary_callable_unary(self):
        dataset = tabular.from_data(
            (('a', lambda i: i * i), ('b', lambda i: -i)), size=10)

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert dataset.keys == ('a', 'b')
        assert dataset.mode == tuple

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 9], [-1, -3]))
        assert isinstance(output[0], list)
        assert isinstance(output[1], list)

    def test_tuple_callable_unary_callable_unary_without_size(self):
        with pytest.raises(ValueError):
            tabular.from_data((('a', lambda i: i * i), ('b', lambda i: -i)))

    def test_dict_array_list(self):
        dataset = tabular.from_data(
            {'a': np.arange(10), 'b': [2, 7, 1, 8, 4, 5, 9, 0, 3, 6]})

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert set(dataset.keys) == {'a', 'b'}
        assert dataset.mode == dict

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, {'a': [1, 3], 'b': [7, 8]})
        assert isinstance(output['a'], np.ndarray)
        assert isinstance(output['b'], list)

    def test_dict_array_callable_unary(self):
        dataset = tabular.from_data({'a': np.arange(10), 'b': lambda i: i * i})

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert set(dataset.keys) == {'a', 'b'}
        assert dataset.mode == dict

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, {'a': [1, 3], 'b': [1, 9]})
        assert isinstance(output['a'], np.ndarray)
        assert isinstance(output['b'], list)

    def test_dict_array_callable_tuple(self):
        dataset = tabular.from_data(
            {'a': np.arange(10), ('b', 'c'): lambda i: (i * i, -i)})

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert set(dataset.keys) == {'a', 'b', 'c'}
        assert dataset.mode == dict

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(
            output, {'a': [1, 3], 'b': [1, 9], 'c': [-1, -3]})
        assert isinstance(output['a'], np.ndarray)
        assert isinstance(output['b'], list)
        assert isinstance(output['c'], list)

    def test_dict_array_callable_dict(self):
        dataset = tabular.from_data(
            {'a': np.arange(10), ('b', 'c'): lambda i: {'b': i * i, 'c': -i}})

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert set(dataset.keys) == {'a', 'b', 'c'}
        assert dataset.mode == dict

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(
            output, {'a': [1, 3], 'b': [1, 9], 'c': [-1, -3]})
        assert isinstance(output['a'], np.ndarray)
        assert isinstance(output['b'], list)
        assert isinstance(output['c'], list)

    def test_dict_callable_unary_callable_unary(self):
        dataset = tabular.from_data(
            {'a': lambda i: i * i, 'b': lambda i: -i}, size=10)

        assert isinstance(dataset, ppe.dataset.TabularDataset)
        assert len(dataset) == 10
        assert set(dataset.keys) == {'a', 'b'}

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, {'a': [1, 9], 'b': [-1, -3]})
        assert isinstance(output['a'], list)
        assert isinstance(output['b'], list)

    def test_dict_callable_unary_callable_unary_without_size(self):
        with pytest.raises(ValueError):
            tabular.from_data(({'a': lambda i: i * i, 'b': lambda i: -i}))

    def test_unique(self):
        dataset_a = tabular.from_data(np.arange(10))
        dataset_b = tabular.from_data(np.arange(10))
        assert dataset_a != dataset_b
