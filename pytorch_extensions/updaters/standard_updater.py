import six

import torch
from pytorch_extensions import _updater
import pytorch_extensions.convert as convert


class StandardUpdater(_updater.Updater):

    """StandardUpdater(\
iterator, optimizer, converter=convert.concat_examples, device=None, \
loss_func=None, loss_scale=None, auto_new_epoch=True, *, input_device=None)

    Standard implementation of Updater.

    This is the standard implementation of :class:`~chainer.training.Updater`.
    It accepts one or more training datasets and one or more optimizers.
    The default update routine assumes that there is only one training dataset
    and one optimizer. Users can override this update routine by inheriting
    this class and overriding the :meth:`update_core` method. Each batch is
    converted to input arrays by :func:`chainer.dataset.concat_examples` by
    default, which can also be manually set by ``converter`` argument.

    Args:
        iterator: Dataset iterator for the training dataset. It can also be a
            dictionary that maps strings to iterators.
            If this is just an iterator, then the
            iterator is registered by the name ``'main'``.
        optimizer: Optimizer to update parameters. It can also be a dictionary
            that maps strings to optimizers.
            If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter: Converter function to build input arrays. Each batch
            extracted by the main iterator and the ``device`` option are passed
            to this function. :func:`chainer.dataset.concat_examples` is used
            by default.
        device(device specifier): Device to which the model is sent.
            If ``None``, the device of the model will stay unchanged.
        loss_func: Loss function. The target link of the main optimizer is used
            by default.
        loss_scale (float): Loss scaling factor. Loss scaling is a usefull
            technique to mitigate vanishing gradient issue that tends to happen
            when low precision data type like float16 is used during training.
            If you set loss scaling factor, gradients of loss values are to be
            multiplied by the factor before backprop starts. The factor is
            propagated to whole gradients in a computational graph along the
            backprop. The gradients of parameters are divided by the factor
            just before the parameters are to be updated.
        auto_new_epoch (bool): If ``True``,
            :meth:`~chainer.Optimizer.new_epoch` of the main optimizer is
            automatically called when the ``is_new_epoch`` attribute of the
            main iterator is ``True``.
        input_device (device specifier):
            Device to which the training data is sent.
            If ``input_device`` is omitted, it will match the ``device``
            argument.

    Attributes:
        converter: Converter function.
        loss_func: Loss function. If it is ``None``, the target link of the
                   main optimizer is used instead.
        device: Device to which the model is sent.
        input_device: Device to which the training data is sent.
        iteration: Current number of completed updates.
        auto_new_epoch: If ``True``, :meth:`~chainer.Optimizer.new_epoch` is
            automatically called by :meth:`update_core`. In this case, the
            :attr:`~chainer.Optimizer.use_auto_new_epoch` attribute of each
            optimizer is also set to ``True``. If :meth:`update_core` is
            overridden, the implementation should correctly call
            :meth:`~chainer.Optimizer.new_epoch` of each optimizer.

    """

    def __init__(self, iterator, models, optimizers,
                 converter=convert.transfer_data,
                 device=None, loss_func=None, loss_scale=None,
                 auto_new_epoch=True, **kwargs):

        input_device = kwargs.get('input_device')

        if device is not None:
            device = torch.device(device)

        # input_device falls back to device
        if input_device is None:
            input_device = device
        else:
            input_device = torch.device(input_device)

        if isinstance(iterator, torch.utils.data.DataLoader):
            # This is a trick for next() call to work
            self.iterator = iter(iterator)
            self._epoch_size = len(iterator)
            iterator = {'main': iterator}

        self._iterators = iterator

        if not isinstance(optimizers, dict):
            optimizers = {'main': optimizers}
        self._optimizers = optimizers

        if not isinstance(models, dict):
            models = {'main': models}
        self._models = models
        self.converter = converter
        self.loss_func = loss_func
        self.iteration = 0
        self._device = device
        self._input_device = input_device

        self.loss_scale = loss_scale
        if loss_scale is not None:
            raise ValueError('loss_scale is not supported')

        self.auto_new_epoch = auto_new_epoch

    @property
    def device(self):
        return self._device

    @property
    def input_device(self):
        return self._input_device

    @property
    def epoch(self):
        return self.iteration // self._epoch_size

    @property
    def epoch_detail(self):
        return self.iteration / self._epoch_size

    @property
    def is_before_training(self):
        return self.iteration == 0

    def finalize(self):
        """Finalizes the updater object.

        This method calls the `finalize` method of each iterator that
        this updater has.
        It is called at the end of training loops.

        """
        pass
        # for iterator in six.itervalues(self._iterators):
        #     iterator.finalize()

    def get_optimizer(self, name):
        """Gets the optimizer of given name.

        Args:
            name (str): Name of the optimizer.

        Returns:
            ~chainer.Optimizer: Corresponding optimizer.

        """
        return self._optimizers[name]

    def get_all_optimizers(self):
        """Gets a dictionary of all optimizers for this updater.

        Returns:
            dict: Dictionary that maps names to optimizers.

        """
        return dict(self._optimizers)

    def get_all_models(self):
        """Gets a dictionary of all models for this updater.

        Returns:
            dict: Dictionary that maps names to models.

        """
        return dict(self._models)

    def get_iterator(self, name):
        """Gets the dataset iterator of given name.

        Args:
            name (str): Name of the dataset iterator.

        Returns:
            ~chainer.dataset.Iterator: Corresponding dataset iterator.

        """
        return self._iterators[name]

    def update(self):
        """Updates the parameters of the target model.

        This method implements an update formula for the training task,
        including data loading, forward/backward computations, and actual
        updates of parameters.

        This method is called once at each iteration of the training loop.

        """
        self.update_core()
        self.iteration += 1

    def update_core(self):
        # Reset the iterator on StopIteration
        try:
            batch = self.iterator.next()
        except StopIteration:
            self.iterator = iter(self._iterators['main'])
            batch = self.iterator.next()
        # This is the function moving the actual data
        # Create a local one for this
        in_arrays = convert._call_converter(
            self.converter, batch, self.input_device)
        optimizer = self._optimizers['main']
        loss_func = self.loss_func or self._models['main']
        optimizer.zero_grad()
        # Loss should be reported in the model itself
        loss = loss_func(*in_arrays)
        loss.backward()
        optimizer.step()

    def serialize(self, serializer):
        # TODO
        """Serializes the current state of the updater object."""
        for name, iterator in six.iteritems(self._iterators):
            iterator.serialize(serializer['iterator:' + name])

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.serialize(serializer['optimizer:' + name])
            optimizer.target.serialize(serializer['model:' + name])

        self.iteration = serializer('iteration', self.iteration)

    def state_dict(self):
        to_save = {}
        # Save manager status ?
        to_save['models'] = {name: self._models[name].state_dict()
                             for name in self._models}
        to_save['optimizers'] = {name: self._optimizers[name].state_dict()
                                 for name in self._optimizers}
        to_save['iteration'] = self.iteration
        return to_save

    def load_state_dict(self, to_load):
        self.iteration = to_load['iteration']
        for name in self._models:
            self._models[name].load_state_dict(to_load['models'][name])

        for name in self._optimizers:
            self._optimizers[name].load_state_dict(to_load['optimizers'][name])
