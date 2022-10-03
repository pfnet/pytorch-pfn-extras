from typing import Any, Dict, Optional, Tuple, Union
import warnings

import numpy
import torch

from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


matplotlib: Any = None
_available: Optional[bool] = None
_plot_color: Any = None
_plot_color_trans: Any = None
_plot_common_kwargs: Any = None


def percentile(a: torch.Tensor, q: Union[float, Tuple[float, ...]], axis: int) -> Any:
    # fallback to numpy
    return torch.Tensor(
        numpy.percentile(a.cpu().numpy(), q, axis))  # type: ignore[no-untyped-call]


def matplotlib_savefun(target: Tuple[Any, Any], file_o: Any) -> None:
    fig, plt = target
    fig.savefig(file_o)
    fig.clf()
    plt.close(fig)


def _try_import_matplotlib() -> None:
    global matplotlib, _available
    global _plot_color, _plot_color_trans, _plot_common_kwargs
    try:
        import matplotlib
        _available = True
    except ImportError:
        _available = False

    if _available:
        if hasattr(matplotlib.colors, 'to_rgba'):
            _to_rgba = matplotlib.colors.to_rgba
        else:
            # For matplotlib 1.x
            _to_rgba = matplotlib.colors.ColorConverter().to_rgba
        _plot_color = _to_rgba('#1f77b4')  # C0 color
        _plot_color_trans = _plot_color[:3] + (0.2,)  # apply alpha
        _plot_common_kwargs = {
            'alpha': 0.2, 'linewidth': 0, 'color': _plot_color_trans}


def _check_available() -> None:
    if _available is None:
        _try_import_matplotlib()

    if not _available:
        warnings.warn('matplotlib is not installed on your environment, '
                      'so nothing will be plotted at this time. '
                      'Please install matplotlib to plot figures.\n\n'
                      '  $ pip install matplotlib\n')


def _unpack_variables(x: Any, memo: Any = None) -> Any:
    if memo is None:
        memo = ()
    if isinstance(x, torch.Tensor):
        memo += (x,)
    elif isinstance(x, torch.nn.Module):
        memo += tuple(x.parameters())
    elif isinstance(x, (list, tuple)):
        for xi in x:
            memo += _unpack_variables(xi)
    return memo


class Reservoir:

    """Reservoir sample with a fixed sized buffer."""

    def __init__(
            self,
            size: int,
            data_shape: Tuple[int, ...],
            dtype: Any = numpy.float32,
    ) -> None:
        self.size = size
        self.data = numpy.zeros((size,) + data_shape, dtype=dtype)
        self.idxs = numpy.zeros((size,), dtype=numpy.int32)
        self.counter = 0

    def add(self, x: Any, idx: Any = None) -> None:
        if self.counter < self.size:
            self.data[self.counter] = x
            self.idxs[self.counter] = idx or self.counter
        elif self.counter >= self.size and \
                numpy.random.random() < self.size / float(self.counter + 1):
            i = numpy.random.randint(self.size)
            self.data[i] = x
            self.idxs[i] = idx or self.counter
        self.counter += 1

    def get_data(self) -> Tuple[Any, Any]:
        idxs = self.idxs[:min(self.counter, self.size)]
        sorted_args = numpy.argsort(idxs)
        return idxs[sorted_args], self.data[sorted_args]


class Statistician:

    """Helper to compute basic NumPy-like statistics."""

    def __init__(
            self,
            collect_mean: bool,
            collect_std: bool,
            percentile_sigmas: Union[float, Tuple[float, ...]],
    ) -> None:
        self.collect_mean = collect_mean
        self.collect_std = collect_std
        self.percentile_sigmas = percentile_sigmas

    def __call__(self, x: Any, axis: Any = 0, dtype: Any = None) -> Dict[str, Any]:
        if axis is None:
            axis = tuple(range(x.ndim))
        elif not isinstance(axis, (tuple, list)):
            axis = axis,

        return self.collect(x, axis)

    def collect(self, x: Any, axis: int) -> Dict[str, Any]:
        out = dict()

        if self.collect_mean:
            out['mean'] = x.mean(axis=axis)

        if self.collect_std:
            out['std'] = x.std(axis=axis)

        if self.percentile_sigmas:
            p = percentile(x, self.percentile_sigmas, axis=axis)
            out['percentile'] = p

        return out


class VariableStatisticsPlot(extension.Extension):

    """__init__(\
targets, max_sample_size=1000, report_data=True, report_grad=True, \
plot_mean=True, plot_std=True, \
percentile_sigmas=(0, 0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87, 100), \
trigger=(1, 'epoch'), filename='statistics.png', figsize=None, marker=None, \
grid=True)

    An extension to plot statistics for :class:`~torch.Tensor`\\s.


    This extension collects statistics for a single :class:`torch.Tensor`,
    a list of :class:`torch.Tensor`\\s or similarly a single or a list of
    :class:`torch.nn.Module`\\s containing one or more
    :class:`torch.Tensor`\\s.  In case multiple :class:`torch.Tensor`\\s
    are found, the means are computed. The collected statistics are plotted
    and saved as an image in the directory specified by the :class:`Manager`.

    Statistics include mean, standard deviation and percentiles.

    This extension uses reservoir sampling to preserve memory, using a fixed
    size running sample. This means that collected items in the sample are
    discarded uniformly at random when the number of items becomes larger
    than the maximum sample size, but each item is expected to occur in the
    sample with equal probability.

    Args:
        targets (:class:`torch.Tensor`, :class:`torch.nn.Module`
             or list of either): Parameters for which statistics are collected.
        max_sample_size (int):
            Maximum number of running samples.
        report_data (bool):
            If ``True``, data (e.g. weights) statistics are plotted.  If
            ``False``, they are neither computed nor plotted.
        report_grad (bool):
            If ``True``, gradient statistics are plotted. If ``False``, they
            are neither computed nor plotted.
        plot_mean (bool):
            If ``True``, means are plotted.  If ``False``, they are
            neither computed nor plotted.
        plot_std (bool):
            If ``True``, standard deviations are plotted.  If ``False``, they
            are neither computed nor plotted.
        percentile_sigmas (float or tuple of floats):
            Percentiles to plot in the range :math:`[0, 100]`.
        trigger:
            Trigger that decides when to save the plots as an image.  This is
            distinct from the trigger of this extension itself. If it is a
            tuple in the form ``<int>, 'epoch'`` or ``<int>, 'iteration'``, it
            is passed to :class:`IntervalTrigger`.
        filename (str):
            Name of the output image file under the output directory.
            For historical reasons ``file_name`` is also accepted as an alias
            of this argument.
        figsize (tuple of int):
            Matlotlib ``figsize`` argument that specifies the size of the
            output image.
        marker (str):
            Matplotlib ``marker`` argument that specified the marker style of
            the plots.
        grid (bool):
            Matplotlib ``grid`` argument that specifies whether grids are
            rendered in in the plots or not.
        writer (writer object, optional): must be callable.
            object to dump the log to. If specified, it needs to have a correct
            `savefun` defined. The writer can override the save location in
            the :class:`pytorch_pfn_extras.training.ExtensionsManager` object
    """

    def __init__(
            self,
            targets: Any,
            max_sample_size: int = 1000,
            report_data: bool = True,
            report_grad: bool = True,
            plot_mean: bool = True,
            plot_std: bool = True,
            percentile_sigmas: Union[float, Tuple[float, ...]] = (
                0, 0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87, 100),
            trigger: trigger_module.TriggerLike = (1, 'epoch'),
            filename: Optional[str] = None,
            figsize: Optional[Tuple[int, ...]] = None,
            marker: Optional[str] = None,
            grid: bool = True,
            **kwargs: Any,
    ):

        _check_available()

        file_name = kwargs.get('file_name', 'statistics.png')
        if filename is None:
            filename = file_name
        del file_name  # avoid accidental use

        self._vars = _unpack_variables(targets)
        if not self._vars:
            raise ValueError(
                'Need at least one variables for which to collect statistics.'
                '\nActual: 0 <= 0')

        if not any((plot_mean, plot_std, bool(percentile_sigmas))):
            raise ValueError('Nothing to plot')

        self._keys = []
        if report_data:
            self._keys.append('data')
        if report_grad:
            self._keys.append('grad')

        self._report_data = report_data
        self._report_grad = report_grad

        self._statistician = Statistician(
            collect_mean=plot_mean, collect_std=plot_std,
            percentile_sigmas=percentile_sigmas)

        self._plot_mean = plot_mean
        self._plot_std = plot_std
        self._plot_percentile = bool(percentile_sigmas)

        self._trigger = trigger_module.get_trigger(trigger)
        self._filename = filename
        self._figsize = figsize
        self._marker = marker
        self._grid = grid
        self._writer = kwargs.get('writer', None)

        if not self._plot_percentile:
            n_percentile = 0
        else:
            if not isinstance(percentile_sigmas, (list, tuple)):
                n_percentile = 1  # scalar, single percentile
            else:
                n_percentile = len(percentile_sigmas)
        self._data_shape = (
            len(self._keys), int(plot_mean) + int(plot_std) + n_percentile)
        self._samples = Reservoir(max_sample_size, data_shape=self._data_shape)

    @staticmethod
    def available() -> bool:
        _check_available()
        assert _available is not None
        return _available

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        if _available:
            # Dynamically import pyplot to call matplotlib.use()
            # after importing pytorch_pfn_extras.training.extensions
            import matplotlib.pyplot as plt
        else:
            return

        stats = numpy.zeros(self._data_shape, dtype=numpy.float32)
        for i, k in enumerate(self._keys):
            xs = []
            for var in self._vars:
                x = getattr(var, k, None)
                if x is not None:
                    xs.append(x.flatten())
            if xs:
                stat_dict = self._statistician(torch.cat(xs, dim=0), axis=0)
                stat_list = []
                if self._plot_mean:
                    stat_list.append(
                        numpy.atleast_1d(stat_dict['mean'].cpu().numpy()))
                if self._plot_std:
                    stat_list.append(
                        numpy.atleast_1d(stat_dict['std'].cpu().numpy()))
                if self._plot_percentile:
                    stat_list.append(
                        numpy.atleast_1d(stat_dict['percentile']))
                stats[i] = numpy.concatenate(  # type: ignore[no-untyped-call]
                    stat_list, axis=0)

        self._samples.add(stats, idx=manager.iteration)

        if self._trigger(manager):
            self.save_plot_using_module(plt, manager)

    def save_plot_using_module(
            self,
            plt: Any,
            manager: ExtensionsManagerProtocol,
    ) -> None:
        nrows = int(self._plot_mean or self._plot_std) \
            + int(self._plot_percentile)
        ncols = len(self._keys)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=self._figsize, sharex=True)

        if not isinstance(axes, numpy.ndarray):  # single subplot
            axes = numpy.asarray([axes])
        if nrows == 1:
            axes = axes[None, :]
        elif ncols == 1:
            axes = axes[:, None]
        assert axes.ndim == 2

        idxs, data = self._samples.get_data()

        # Offset to access percentile data from `data`
        offset = int(self._plot_mean) + int(self._plot_std)
        n_percentile = data.shape[-1] - offset
        n_percentile_mid_floor = n_percentile // 2
        n_percentile_odd = n_percentile % 2 == 1

        writer = manager.writer if self._writer is None else self._writer

        for col in range(ncols):
            row = 0
            ax = axes[row, col]
            ax.set_title(self._keys[col])  # `data` or `grad`

            if self._plot_mean or self._plot_std:
                if self._plot_mean and self._plot_std:
                    ax.errorbar(
                        idxs, data[:, col, 0], data[:, col, 1],
                        color=_plot_color, ecolor=_plot_color_trans,
                        label='mean, std', marker=self._marker)
                else:
                    if self._plot_mean:
                        label = 'mean'
                    elif self._plot_std:
                        label = 'std'
                    ax.plot(
                        idxs, data[:, col, 0], color=_plot_color, label=label,
                        marker=self._marker)
                row += 1

            if self._plot_percentile:
                ax = axes[row, col]
                for i in range(n_percentile_mid_floor + 1):
                    if n_percentile_odd and i == n_percentile_mid_floor:
                        # Enters at most once per sub-plot, in case there is
                        # only a single percentile to plot or when this
                        # percentile is the mid percentile and the number of
                        # percentiles are odd
                        ax.plot(
                            idxs, data[:, col, offset + i], color=_plot_color,
                            label='percentile', marker=self._marker)
                    else:
                        if i == n_percentile_mid_floor:
                            # Last percentiles and the number of all
                            # percentiles are even
                            label = 'percentile'
                        else:
                            label = '_nolegend_'
                        ax.fill_between(
                            idxs,
                            data[:, col, offset + i],
                            data[:, col, -i - 1],
                            label=label,
                            **_plot_common_kwargs)
                    ax.set_xlabel('iteration')

        for ax in axes.ravel():
            ax.legend()
            if self._grid:
                ax.grid()
                ax.set_axisbelow(True)

        writer(self._filename, manager.out, (fig, plt),  # type: ignore
               savefun=matplotlib_savefun)

    def finalize(self, manager: ExtensionsManagerProtocol) -> None:
        if self._writer is not None:
            self._writer.finalize()
