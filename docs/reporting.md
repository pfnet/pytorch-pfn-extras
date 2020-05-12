# Reporting

`reporting.Reporter` is used to collect values that users want to watch.
The reporter object holds a mapping from value names to the actually observed values. We call this mapping observations.

When a value is passed to the reporter, an object called observer can be optionally attached. In this case, the name of the observer is added as the prefix of the value name. The observer name should be registered beforehand.

```python
import pytorch_pfn_extras as ppe

reporter = ppe.reporting.Reporter()
observer = object()
reporter.add_observer('my_observer', observer)
observation = {}

with reporter.scope(observation):
    reporter.report({'x': 1}, observer)

print(observation)
# outputs: {'my_observer/x': 1}
```

There is also a global API to add values:

```python
import pytorch_pfn_extras as ppe

reporter = ppe.reporting.Reporter()
observer = object()
reporter.add_observer('my_observer', observer)

observation = {}
with reporter:
    with ppe.reporting.report_scope(observation):
         ppe.reporting.report({'x': 1}, observer)

print(observation)
# outputs: {'my_observer/x': 1}
```

The most important application of Reporter is to report observed values from different parts of the model in the training
and validation procedures. `ExtensionsManager` objects hold their own `Reporter` object with the parameters of the target
module registered as observers. `report()` can be used inside the modules to report the observed values (e.g., training loss,
accuracy, activation statistics, etc.).
