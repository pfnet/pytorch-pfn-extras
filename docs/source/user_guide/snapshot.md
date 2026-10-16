# Distributed Snapshot

To take snapshots when using `torch.distributed` the only needed step is to
provide the `saver_rank` keyword argument to the regular snapshot extension.

```python
# saver_rank is the MPI rank which will write the actual snapshot.
snapshot = extensions.snapshot(saver_rank=saver_rank)
```

To resume the training, snapshots are loaded in every worker by using the 
`ExtensionsManager.load_state_dict` method, or the `extensions.snapshot`
`autoload` keyword argument.

Autoloading requires model state keys to match by default. If a model's
registered buffers or parameters have changed, use
`extensions.snapshot(autoload=True, strict=False)` (or pass `strict=False`
to `snapshot_object`) to allow missing or unexpected model keys. Matching
values are restored; missing values keep their current initialization.
Tensor shape mismatches still raise an error.

The same option is available when loading a manager explicitly:
`manager.load_state_dict(state, strict=False)`. It applies only to models;
optimizers, extensions, and other state objects keep their usual loading
behavior. Custom model loaders must accept the `strict` keyword to use
non-strict loading.
