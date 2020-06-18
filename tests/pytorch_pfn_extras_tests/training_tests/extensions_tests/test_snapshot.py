import glob
import itertools
import os
import tempfile
import time
from unittest import mock

import torch
import pytest

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import extensions
from pytorch_pfn_extras.training.extensions._snapshot import (
    _find_latest_snapshot, _find_snapshot_files, _find_stale_snapshots)
from pytorch_pfn_extras import writing


def get_trainer(*, out_dir, state_to_load=None):
    model_state_dict = {}
    optimizer_state_dict = {}
    models = {'main': _StateDictModel(state_dict=model_state_dict)}
    optimizers = {'main': _StateDictObj(state_dict=optimizer_state_dict)}
    epochs = 10  # FIXME
    return training.ExtensionsManager(
        models, optimizers, epochs,
        iters_per_epoch=10,
        out_dir=out_dir)


class _StateDictObj():
    def __init__(self, *, state_dict=None):
        super().__init__()
        self.called_load_state_dict = 0
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self.called_load_state_dict += 1
        self._state_dict = state_dict


class _StateDictModel(_StateDictObj, torch.nn.Module):
    def forward(self, *args):
        pass


def test_call():
    t = mock.MagicMock()
    c = mock.MagicMock(side_effect=[True, False])
    w = mock.MagicMock()
    snapshot = extensions.snapshot(target=t, condition=c, writer=w)
    trainer = mock.MagicMock()
    snapshot(trainer)
    snapshot(trainer)

    assert c.call_count == 2
    assert w.call_count == 1


def test_savefun_and_writer_exclusive():
    # savefun and writer arguments cannot be specified together.
    def savefun(*args, **kwargs):
        assert False
    writer = writing.SimpleWriter()
    with pytest.raises(TypeError):
        extensions.snapshot(savefun=savefun, writer=writer)

    trainer = mock.MagicMock()
    with pytest.raises(TypeError):
        extensions.snapshot_object(trainer, savefun=savefun, writer=writer)


@pytest.fixture(scope='function')
def remover():
    yield
    if os.path.exists('myfile.dat'):
        os.remove('myfile.dat')


def test_save_file(remover):
    trainer = get_trainer(out_dir='.')
    trainer._done = True
    w = writing.SimpleWriter()
    snapshot = extensions.snapshot_object(trainer, 'myfile.dat',
                                          writer=w)
    snapshot(trainer)

    assert os.path.exists('myfile.dat')


def test_multi_target(remover):
    trainer = get_trainer(out_dir='.')
    trainer._done = True
    other_state_dict = {'test': True}
    other = _StateDictObj(state_dict=other_state_dict)
    w = ppe.writing.SimpleWriter()
    target = {'trainer': trainer, 'other': other}
    snapshot = extensions.snapshot_object(target, 'myfile.dat',
                                          writer=w)
    snapshot(trainer)

    assert os.path.exists('myfile.dat')
    # Load the snapshot and verify it
    state = torch.load('myfile.dat')
    new_trainer = get_trainer(out_dir='.')
    new_other = _StateDictObj(state_dict={})
    new_trainer.load_state_dict(state['trainer'])
    new_other.load_state_dict(state['other'])
    assert new_trainer.state_dict() == trainer.state_dict()
    assert new_other.state_dict() == other_state_dict


def test_multi_target_autoload(remover):
    trainer = get_trainer(out_dir='.')
    trainer._done = True
    other_state_dict = {'test': True}
    other = _StateDictObj(state_dict=other_state_dict)
    w = ppe.writing.SimpleWriter()
    target = {'trainer': trainer, 'other': other}
    snapshot = extensions.snapshot_object(target, 'myfile.dat',
                                          writer=w)
    snapshot(trainer)

    assert os.path.exists('myfile.dat')
    new_trainer = get_trainer(out_dir='.')
    new_other = _StateDictObj(state_dict={})

    target = {'trainer': new_trainer, 'other': new_other}
    snapshot2 = extensions.snapshot_object(target, 'myfile.dat',
                                           autoload=True)
    # Load the snapshot and verify it
    snapshot2.initialize(new_trainer)
    assert new_trainer.state_dict() == trainer.state_dict()
    assert new_other.state_dict() == other_state_dict


def test_clean_up_tempdir(remover):
    trainer = get_trainer(out_dir='.')
    trainer._done = True
    snapshot = extensions.snapshot_object(trainer, 'myfile.dat')
    snapshot(trainer)

    left_tmps = [fn for fn in os.listdir('.')
                 if fn.startswith('tmpmyfile.dat')]
    assert len(left_tmps) == 0


def test_on_error():
    # Will fail when accesing the dummy optimizer
    optimizers = {'main': object()}
    trainer = training.ExtensionsManager(
        {}, optimizers, 1,
        iters_per_epoch=1,
        out_dir='.')
    filename = 'myfile-deadbeef.dat'

    snapshot = extensions.snapshot_object(trainer, filename,
                                          snapshot_on_error=True)
    trainer.extend(snapshot)
    assert not os.path.exists(filename)
    with pytest.raises(AttributeError):
        with trainer.run_iteration():
            pass
    assert not os.path.exists(filename)


@pytest.fixture(scope='function')
def path():
    with tempfile.TemporaryDirectory() as t_path:
        yield t_path


@pytest.mark.parametrize('fmt', [
    'snapshot_iter_{}',
    'snapshot_iter_{}.npz',
    '{}_snapshot_man_suffix.npz',
])
def test_find_snapshot_files(fmt, path):
    files = (fmt.format(i) for i in range(1, 100))
    noise = ('dummy-foobar-iter{}'.format(i) for i in range(10, 304))
    noise2 = ('tmpsnapshot_iter_{}'.format(i) for i in range(10, 304))

    for file in itertools.chain(noise, files, noise2):
        file = os.path.join(path, file)
        open(file, 'w').close()

    writer = ppe.writing.SimpleWriter()
    snapshot_files = _find_snapshot_files(fmt, path, writer.fs)

    expected = sorted([fmt.format(i) for i in range(1, 100)])
    assert len(snapshot_files) == 99
    timestamps, snapshot_files = zip(*snapshot_files)
    assert expected == sorted(list(snapshot_files))


@pytest.mark.parametrize('fmt', [
    'snapshot_iter_{}',
    'snapshot_iter_{}.npz',
    '{}_snapshot_man_suffix.npz',
])
def test_find_latest_snapshot(fmt, path):
    files = [fmt.format(i) for i in range(1, 100)]
    base_timestamp = time.time()

    for i, file in enumerate(files):
        file = os.path.join(path, file)
        open(file, 'w').close()

        # mtime resolution of some filesystems e.g. ext3 or HFS+
        # is a second and thus snapshot files such as
        # ``snapshot_iter_9`` and ``snapshot_iter_99`` may have
        # same timestamp if it does not have enough interval
        # between file creation. As current autosnapshot does not
        # uses integer knowledge, timestamp is intentionally
        # modified here. This comment also applies to other tests
        # in this file on snapshot freshness.
        t = base_timestamp + i
        os.utime(file, (t, t))
    writer = ppe.writing.SimpleWriter()
    assert fmt.format(99) == _find_latest_snapshot(fmt, path, writer.fs)


@pytest.mark.parametrize('fmt', [
    'snapshot_iter_{}_{}',
    'snapshot_iter_{}_{}.npz',
    '{}_snapshot_man_{}-suffix.npz',
    'snapshot_iter_{}.{}',
])
def test_find_snapshot_files2(fmt, path):
    files = (fmt.format(i*10, j*10) for i, j
             in itertools.product(range(0, 10), range(0, 10)))
    noise = ('tmpsnapshot_iter_{}.{}'.format(i, j)
             for i, j in zip(range(10, 304), range(10, 200)))

    for file in itertools.chain(noise, files):
        file = os.path.join(path, file)
        open(file, 'w').close()

    writer = ppe.writing.SimpleWriter()
    snapshot_files = _find_snapshot_files(fmt, path, writer.fs)

    expected = [fmt.format(i*10, j*10)
                for i, j in itertools.product(range(0, 10), range(0, 10))]

    timestamps, snapshot_files = zip(*snapshot_files)
    expected.sort()
    snapshot_files = sorted(list(snapshot_files))
    assert expected == snapshot_files


@pytest.mark.parametrize('length_retain', [
    (100, 30), (10, 30), (1, 1000),
    (1000, 1), (1, 1), (1, 3), (2, 3),
])
def test_find_stale_snapshot(length_retain, path):
    length, retain = length_retain
    fmt = 'snapshot_iter_{}'
    files = [fmt.format(i) for i in range(0, length)]
    base_timestamp = time.time() - length * 2

    for i, file in enumerate(files):
        file = os.path.join(path, file)
        open(file, 'w').close()

        # Same comment applies here. See comment in ``TestFindSnapshot``
        t = base_timestamp + i
        os.utime(file, (t, t))

    writer = ppe.writing.SimpleWriter()
    stale = list(_find_stale_snapshots(fmt, path, retain, writer.fs))
    assert max(length-retain, 0) == len(stale)
    expected = [fmt.format(i) for i in range(0, max(length-retain, 0))]
    assert expected == stale


def test_remove_stale_snapshots(path):
    fmt = 'snapshot_iter_{.iteration}'
    retain = 3
    snapshot = extensions.snapshot(filename=fmt, n_retains=retain,
                                   autoload=False)

    trainer = get_trainer(out_dir=path)
    trainer.extend(snapshot, trigger=(1, 'iteration'), priority=2)

    class TimeStampUpdater():
        t = time.time() - 100
        name = 'ts_updater'
        priority = 1  # This must be called after snapshot taken

        def __call__(self, _trainer):
            filename = os.path.join(_trainer.out, fmt.format(_trainer))
            self.t += 1
            # For filesystems that does low timestamp precision
            os.utime(filename, (self.t, self.t))

    trainer.extend(TimeStampUpdater(), trigger=(1, 'iteration'))
    for _ in range(10):
        with trainer.run_iteration():
            pass
    assert 10 == trainer.iteration

    pattern = os.path.join(trainer.out, "snapshot_iter_*")
    found = [os.path.basename(path) for path in glob.glob(pattern)]
    assert retain == len(found)
    found.sort()
    # snapshot_iter_(8, 9, 10) expected
    expected = ['snapshot_iter_{}'.format(i) for i in range(8, 11)]
    expected.sort()
    assert expected == found

    trainer2 = get_trainer(
        out_dir=path, state_to_load=trainer.state_dict())
    snapshot2 = extensions.snapshot(filename=fmt, autoload=True)
    # Just making sure no error occurs
    snapshot2.initialize(trainer2)


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self._wrapper_module = model
        self.accessed = False

    def wrapper_module(self):
        self.accessed = True
        return self._wrapper_module


def test_model_transformations(path):
    model_state_dict = object()
    optimizer_state_dict = object()
    max_epochs = 5
    iters_per_epoch = 4
    model = Wrapper(_StateDictModel(state_dict=model_state_dict))
    manager = training.ExtensionsManager(
        model,
        _StateDictObj(state_dict=optimizer_state_dict),
        max_epochs,
        iters_per_epoch=iters_per_epoch,
        out_dir=path,
    )

    snapshot = extensions.snapshot(
        filename='test',
        transform_models=lambda n, x: x.wrapper_module())
    snapshot(manager)

    assert model.accessed

    # Verify that autoload applies the transformation
    to_load = torch.load(os.path.join(path, 'test'))
    trainer = get_trainer(
        out_dir=path, state_to_load=to_load)
    snapshot = extensions.snapshot(
        filename='test', autoload=True,
        autoload_transform_models=lambda n, x: Wrapper(x))
    snapshot.initialize(trainer)
    assert isinstance(trainer._models['main'], Wrapper)
