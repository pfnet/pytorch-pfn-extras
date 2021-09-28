# mypy: ignore-errors

import torch


class CollateAsDict:
    """Creates a collate function that converts inputs to a dict of tensors.

    An instantiated callable object can be feeded to
    :class:`torch.utils.data.DataLoader` as a ``collate_fn`` option.

    Args:
        names (list of str): Names of keys of output dict.
        collate_fn (function): A function preprocesses inputs.
    """

    def __init__(
        self, names,
        collate_fn=torch.utils.data._utils.collate.default_collate,
    ):
        self.names = names
        self.collate_fn = collate_fn

    def __call__(self, *args, **kwargs):
        """Converts inputs the dataset generated to a dictionary of tensors.

        Returns (dict of Tensor):
            A dictionary with keys that specified as ``names`` option, and
            values as input tensors.
        """
        batch = self.collate_fn(*args, **kwargs)
        return {name: v for name, v in zip(self.names, batch)}
