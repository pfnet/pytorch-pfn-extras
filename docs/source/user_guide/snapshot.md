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
