# Config system

Algorithm configurations such as training hyperparameters are serialized as nested mappings often using the YAML format.  
Nested mappings are parsed by `pytorch_pfn_extras.config.Config` that returns a dictionary of objects that are lazily evaluated.
The most notable feature is a special handling of the `type` key, which invokes a function or instantiates an object. The arguments for them are mapped by non-`type` keys.
```python 
import yaml
from pytorch_pfn_extras.config import Config
from torchvision.models.resnet import resnet50

types = {
  'resnet50': resnet50
}
pre_eval_config = yaml.load('''
model:
  type: resnet50
  num_classes: 10
''')
config = Config(pre_eval_config, types)
resnet = config['/model']  # same as resnet50(num_classes=10)
```
`Config` takes a nested mapping and `types`, which is a dictionary of methods and classes. `type` should map to a key in `types`.

The system allows referencing other values in a given nested mapping using a special `@` syntax. The path to a referenced value is represented using `/`s.
```python
types = {
  'get_dataset_n_class': lambda: 10,  # could be an arbitrary function that returns int
  'resnet50': resnet50
} 

pre_eval_config = yaml.load('''
dataset:
  n_class:
    type: get_dataset_n_class
model:
  type: resnet50
  num_classes: '@dataset/n_class'
''')
config = Config(pre_eval_config, types)
resnet = config['/model']  # same as resnet50(num_classes=get_dataset_n_class())
```
Not only can you reference a value, but you can also reference an attribute of another value using `.`s with `@`.
```python

class Dataset(object):
    n_class = 10

types = {
  'resnet50': resnet50,
  'Dataset': Dataset
} 

pre_eval_config = yaml.load('''
dataset:
  train:
    type: Dataset
model:
  type: resnet50
  num_classes: '@/dataset/train.n_class'
''')
config = Config(pre_eval_config, types)
resnet = config['/model']  # same as resnet50(num_classes=Dataset().n_class)
```


### `customize_type`
Configuration can get large, so you may want to simplify it by using default values.
There are two ways to do this.
The first is simply using Python default arguments for functions.
The second is using `customize_type`. This comes in handy when you want to define default values by referencing another value in the config system.
For instance, in the `resnet50` example, if you always define `n_class` at `dataset/n_class`, it is reasonable to set the default value of `num_classes` to the value of `dataset/n_class`.
This could be achieved as below.
Note that `@` notation is not used by `customize_type`.

```python
@customize_type(num_classes='/dataset/n_class')
def my_resnet50(num_classes):
    return resnet50(num_classes=num_classes)

types = {
  'get_dataset_n_class': lambda: 10,  # could be an arbitrary function that returns int
  'resnet50': my_resnet50
}

pre_eval_config = yaml.load('''
dataset:
  n_class:
    type: get_dataset_n_class
model:
  type: resnet50
''')
config = Config(pre_eval_config, types)
resnet = config['/model']  # same as resnet50(num_classes=get_dataset_n_class())
print(resnet.fc.out_features)  # 10
```


# Why this representation

WIP
