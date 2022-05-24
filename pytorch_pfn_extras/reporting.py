import collections
import contextlib
import threading
import types
from typing import (
    Any, Callable, Dict, Generator, List, Mapping, Optional, Sequence,
    Tuple, Type, Union,
)
from typing import overload
import warnings

import numpy
import torch


Scalar = Union[torch.Tensor, numpy.ndarray, numpy.floating, float]
FloatLikeValue = Union[Scalar, float]
Value = Union[Scalar, Callable[[], float]]
Observation = Dict[str, Value]


_thread_local = threading.local()


@overload
def _nograd(value: FloatLikeValue) -> FloatLikeValue:
    ...


@overload
def _nograd(value: Value) -> Value:
    ...


def _nograd(
        value: Union[FloatLikeValue, Value]) -> Union[FloatLikeValue, Value]:
    if isinstance(value, torch.Tensor):
        return value.detach()
    return value


class Reporter:

    """Object to which observed values are reported.

    Reporter is used to collect values that users want to watch. The reporter
    object holds a mapping from value names to the actually observed values.
    We call this mapping `observations`.

    When a value is passed to the reporter, an object called `observer` can be
    optionally attached. In this case, the name of the observer is added as the
    prefix of the value name. The observer name should be registered
    beforehand.

    See the following example:

    >>> from pytorch_pfn_extras.reporting import Reporter, report, report_scope
    >>>
    >>> reporter = Reporter()
    >>> observer = object()  # it can be an arbitrary (reference) object
    >>> reporter.add_observer('my_observer', observer)
    >>> observation = {}
    >>> with reporter.scope(observation):
    ...     reporter.report({'x': 1}, observer)
    ...
    >>> observation
    {'my_observer/x': 1}

    There are also a global API to add values:

    >>> reporter = Reporter()
    >>> observation = {}
    >>> with reporter:
    ...     with report_scope(observation):
    ...         report({'x': 1})
    ...
    >>> observation
    {'x': 1}

    The most important application of Reporter is to report observed values
    from each link or chain in the training and validation procedures.
    and some extensions prepare their own
    Reporter object with the hierarchy of the target module registered as
    observers. We can use :func:`report` function inside any nn.Module
    to report the observed values (e.g., training loss, accuracy, activation
    statistics, etc.).

    Attributes:
        observation: Dictionary of observed values.

    """

    def __init__(self) -> None:
        self._observer_names: Dict[int, str] = {}
        self.observation: Observation = {}

    def __enter__(self) -> None:
        """Makes this reporter object current."""
        _get_reporters().append(self)

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[types.TracebackType],
    ) -> None:
        """Recovers the previous reporter object to the current."""
        _get_reporters().pop()

    @contextlib.contextmanager
    def scope(self, observation: Observation) -> Generator[None, None, None]:
        """Creates a scope to report observed values to ``observation``.

        This is a context manager to be passed to ``with`` statements. In this
        scope, the observation dictionary is changed to the given one.

        It also makes this reporter object current.

        Args:
            observation (dict): Observation dictionary. All observations
                reported inside of the ``with`` statement are written to this
                dictionary.

        """
        old = self.observation
        self.observation = observation
        self.__enter__()
        try:
            yield
        finally:
            self.__exit__(None, None, None)
            self.observation = old

    def add_observer(self, name: str, observer: torch.nn.Module) -> None:
        """Registers an observer of values.

        Observer defines a scope of names for observed values. Values observed
        with the observer are registered with names prefixed by the observer
        name.

        Args:
            name (str): Name of the observer.
            observer: The observer object. Note that the reporter distinguishes
                the observers by their object ids (i.e., ``id(owner)``), rather
                than the object equality.

        """
        self._observer_names[id(observer)] = name

    def add_observers(
            self,
            prefix: str,
            observers: Sequence[Tuple[str, torch.nn.Module]]
    ) -> None:
        """Registers multiple observers at once.

        This is a convenient method to register multiple objects at once.

        Args:
            prefix (str): Prefix of each name of observers.
            observers: Iterator of name and observer pairs.

        """
        for name, observer in observers:
            self._observer_names[id(observer)] = prefix + name

    def report(
            self,
            values: Mapping[str, Value],
            observer: Optional[torch.nn.Module] = None,
    ) -> None:
        """Reports observed values.

        The values are written with the key, prefixed by the name of the
        observer object if given.

        .. note::
           If a value is of type :class:`~torch.Tensor`, the
           variable is copied without preserving the computational graph and
           the new variable object purged from the graph is stored to the
           observer.

        Args:
            values (dict): Dictionary of observed values.
            observer: Observer object. Its object ID is used to retrieve the
                observer name, which is used as the prefix of the registration
                name of the observed value.

        """
        values = {k: _nograd(v) for k, v in values.items()}

        if observer is not None:
            observer_id = id(observer)
            if observer_id not in self._observer_names:
                raise KeyError(
                    'Given observer is not registered to the reporter.')
            observer_name = self._observer_names[observer_id]
            for key, value in values.items():
                name = '%s/%s' % (observer_name, key)
                self.observation[name] = value
        else:
            self.observation.update(values)


def _get_reporters() -> List[Reporter]:
    try:
        reporters: List[Reporter] = _thread_local.reporters
    except AttributeError:
        reporters = _thread_local.reporters = []
    return reporters


def get_current_reporter() -> Reporter:
    """Returns the current reporter object."""
    return _get_reporters()[-1]


def report(
        values: Mapping[str, Value],
        observer: Optional[torch.nn.Module] = None,
) -> None:
    """Reports observed values with the current reporter object.

    Any reporter object can be set current by the ``with`` statement. This
    function calls the :meth:`Reporter.report` method of the current reporter.
    If no reporter object is current, this function does nothing.

    .. admonition:: Example

       The most typical example is a use within `nn.Module`. Suppose that
       a module is registered to the current reporter as an observer (for
       example, the target module of the optimizer is automatically
       registered to the main reporter. We can report
       some values from the link as follows::

          class MyRegressor:
              def __init__(self, predictor):
                  super().__init__(predictor=predictor)

              def __call__(self, x, y):
                  # This chain just computes the mean absolute and squared
                  # errors between the prediction and y.
                  pred = self.predictor(x)
                  abs_error = F.sum(abs(pred - y)) / len(x)
                  loss = F.mean_squared_error(pred, y)

                  # Report the mean absolute and squared errors.
                  reporter.report({
                      'abs_error': abs_error,
                      'squared_error': loss,
                  }, self)

                  return loss

       If the module is named ``'main'`` in the hierarchy these reported values
       are named ``'main/abs_error'`` and ``'main/squared_error'``.

    Args:
        values (dict): Dictionary of observed values.
        observer: Observer object. Its object ID is used to retrieve the
            observer name, which is used as the prefix of the registration name
            of the observed value.

    """
    reporters = _get_reporters()
    if reporters:
        current = reporters[-1]
        current.report(values, observer)


@contextlib.contextmanager
def report_scope(observation: Observation) -> Generator[None, None, None]:
    """Returns a report scope with the current reporter.

    This is equivalent to ``get_current_reporter().scope(observation)``,
    except that it does not make the reporter current redundantly.

    """
    current = _get_reporters()[-1]
    old = current.observation
    current.observation = observation
    yield
    current.observation = old


class Summary:

    """Online summarization of a sequence of scalars.

    Summary computes the statistics of given scalars online.

    """

    def __init__(self) -> None:
        self._x: Scalar = 0.0
        self._x2: Scalar = 0.0
        self._n: Scalar = 0
        self._deferred: List[Tuple[Callable[[], float], Scalar]] = []

    def _add_deferred_values(self) -> None:
        for fn, weight in self._deferred:
            value = fn()
            self.add(value, weight)
        self._deferred.clear()

    def add(self, value: Value, weight: Scalar = 1) -> None:
        """Adds a scalar value.

        Args:
            value: Scalar value to accumulate. It is either a NumPy scalar or
                a zero-dimensional array (on CPU or GPU).
            weight: An optional weight for the value. It is a NumPy scalar or
                a zero-dimensional array (on CPU or GPU).
                Default is 1 (integer).

        """
        if callable(value):
            self._deferred.append((value, weight))
            return

        self._x += weight * value
        self._x2 += weight * value * value
        self._n += weight

    def compute_mean(self) -> Scalar:
        """Computes the mean."""
        self._add_deferred_values()

        x, n = self._x, self._n
        return x / n

    def make_statistics(self) -> Tuple[Scalar, Scalar]:
        """Computes and returns the mean and standard deviation values.

        Returns:
            tuple: Mean and standard deviation values.

        """
        self._add_deferred_values()

        x, n = self._x, self._n
        mean = x / n
        var = self._x2 / n - mean * mean
        if isinstance(var, torch.Tensor):
            return mean, torch.sqrt(var)
        else:
            return mean, numpy.sqrt(var)

    def state_dict(self) -> Dict[str, Any]:
        self._add_deferred_values()
        state = {}
        try:
            # Save the stats as python scalars in order to avoid
            # different device errors when loading them back
            state = {'_x': float(self._x),
                     '_x2': float(self._x2),
                     '_n': int(self._n)}
        except KeyError:
            warnings.warn('The previous statistics are not saved.')
        return state

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        # Casting here is because of backward compatibility
        # Restore previously taken snapshots with autoload
        self._add_deferred_values()
        self._x = float(_nograd(to_load['_x']))
        self._x2 = float(_nograd(to_load['_x2']))
        self._n = int(_nograd(to_load['_n']))

    def __add__(self, other: "Summary") -> "Summary":
        s = Summary()
        s._x = self._x + other._x
        s._x2 = self._x2 + other._x2
        s._n = self._n + other._n
        s._deferred = self._deferred + other._deferred
        return s


class DictSummary:

    """Online summarization of a sequence of dictionaries.

    ``DictSummary`` computes the statistics of a given set of scalars online.
    It only computes the statistics for scalar values and variables of scalar
    values in the dictionaries.

    """

    def __init__(self) -> None:
        self._summaries: Dict[str, Summary] = collections.defaultdict(Summary)

    def add(self, d: Mapping[str, Union[Value, Tuple[Value, Scalar]]]) -> None:
        """Adds a dictionary of scalars.

        Args:
            d (dict): Dictionary of scalars to accumulate. Only elements of
               scalars, zero-dimensional arrays, and variables of
               zero-dimensional arrays are accumulated. When the value
               is a tuple, the second element is interpreted as a weight.

        """
        summaries = self._summaries
        for k, v in d.items():
            w: Scalar = 1
            if isinstance(v, tuple):
                v, w = v
                if not numpy.isscalar(w) and not getattr(w, 'ndim', -1) == 0:
                    raise ValueError(
                        'Given weight to {} was not scalar.'.format(k))
            if callable(v) or numpy.isscalar(v) or getattr(v, 'ndim', -1) == 0:
                summaries[k].add(v, weight=w)

    def compute_mean(self) -> Dict[str, Scalar]:
        """Creates a dictionary of mean values.

        It returns a single dictionary that holds a mean value for each entry
        added to the summary.

        Returns:
            dict: Dictionary of mean values.

        """
        return {name: summary.compute_mean()
                for name, summary in self._summaries.items()}

    def make_statistics(self) -> Dict[str, Scalar]:
        """Creates a dictionary of statistics.

        It returns a single dictionary that holds mean and standard deviation
        values for every entry added to the summary. For an entry of name
        ``'key'``, these values are added to the dictionary by names ``'key'``
        and ``'key.std'``, respectively.

        Returns:
            dict: Dictionary of statistics of all entries.

        """
        stats = {}
        for name, summary in self._summaries.items():
            mean, std = summary.make_statistics()
            stats[name] = mean
            stats[name + '.std'] = std

        return stats

    def state_dict(self) -> Dict[str, Any]:
        return {
            name: summ.state_dict() for name, summ in self._summaries.items()}

    def load_state_dict(self, to_load: Dict[str, Any]) -> None:
        self._summaries.clear()
        for name, summ_state in to_load.items():
            self._summaries[name].load_state_dict(summ_state)

    def __add__(self, other: "DictSummary") -> "DictSummary":
        s1, s2 = self._summaries, other._summaries
        ds = DictSummary()
        for k in sorted(list(set([*s1.keys(), *s2.keys()]))):
            if k not in s1:
                ds._summaries[k] = s2[k]
            elif k not in s2:
                ds._summaries[k] = s1[k]
            else:
                ds._summaries[k] = s1[k] + s2[k]
        return ds
