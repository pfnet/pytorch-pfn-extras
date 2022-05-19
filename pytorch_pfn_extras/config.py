import json
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import reprlib


ConfigKey = Tuple[Union[str, int], ...]
AttrKey = Tuple[Union[str, int], ...]
KeyPair = Tuple[ConfigKey, Optional[AttrKey]]
ConfigType = Any
Loader = Callable[[str], Any]
DumpTrace = Tuple[KeyPair, ...]
LoadTrace = Tuple[Tuple[str, ConfigKey], ...]


def customize_type(**default_kwargs: Any) -> Callable[
        [Callable[..., Any]], Callable[..., Any]]:
    def deco(type_: Callable[..., Any]) -> Callable[..., Any]:
        type_._custom_default_kwargs = default_kwargs  # type: ignore[attr-defined] # NOQA
        return type_
    return deco


class Config(object):

    def __init__(
            self,
            config: Any,
            types: Optional[Mapping[str, Callable[..., Any]]] = None,
    ) -> None:
        self._cache: Dict[KeyPair, Any] = {((), None): config}
        self._types = types or {}

    def __getitem__(self, key: str) -> Any:
        return self._eval(*_parse_key(key, ())[:2], ())

    @classmethod
    def load_path(
            cls,
            path: str,
            *,
            loader: Optional[Loader] = None,
            types: Optional[Mapping[str, Callable[..., Any]]] = None,
    ) -> 'Config':
        if loader is None:
            loader = _json_loader
        return cls(_load(path, loader, ()), types)

    def _eval(
            self,
            config_key: ConfigKey,
            attr_key: Optional[AttrKey],
            trace: DumpTrace,
    ) -> Any:
        if (config_key, attr_key) in self._cache:
            return self._cache[(config_key, attr_key)]

        circular = (config_key, attr_key) in trace
        trace = (*trace, (config_key, attr_key))
        if circular:
            raise RuntimeError(
                'Circular dependency',
                _dump_trace(trace))

        def cache(value: Any) -> Any:
            self._cache[(config_key, attr_key)] = value
            return value

        if attr_key:
            obj = self._eval(config_key, attr_key[:-1], trace)
            try:
                if isinstance(attr_key[-1], str) \
                   and hasattr(obj, attr_key[-1]):
                    return cache(getattr(obj, attr_key[-1]))
                else:
                    return cache(obj[attr_key[-1]])
            except Exception as e:
                e.args = e.args + (
                    '{} not in {} ({})'.format(
                        attr_key[-1],
                        _dump_key(config_key, attr_key[:-1]),
                        reprlib.repr(obj)),
                    _dump_trace(trace))
                raise e

        elif attr_key is None:
            config = self._eval(config_key[:-1], None, trace)
            try:
                return cache(config[config_key[-1]])
            except Exception as e:
                e.args = e.args + (
                    '{} not in {}'.format(
                        config_key[-1],
                        _dump_key(config_key[:-1], None)),
                    _dump_trace(trace))
                raise e

        else:
            config = self._eval(config_key, None, trace)
            if isinstance(config, dict):
                if 'type' in config:
                    try:
                        type_ = self._types[config['type']]
                    except Exception as e:
                        e.args = e.args + (
                            '{} not in types'.format(config['type']),
                            _dump_trace(trace))
                        raise e
                else:
                    type_ = dict

                kwargs = {}
                for k in config.keys():
                    if not k == 'type':
                        kwargs[k] = self._eval((*config_key, k), (), trace)
                for k, v in getattr(
                        type_, '_custom_default_kwargs', {}).items():
                    if k not in kwargs:
                        kwargs[k] = self._eval(
                            *_parse_key(v, config_key)[:2], trace)

                try:
                    return cache(type_(**kwargs))
                except Exception as e:
                    e.args = e.args + (
                        '{} ({}) failed with kwargs {}'.format(
                            config['type'], type_, reprlib.repr(kwargs)),
                        _dump_trace(trace))
                    raise e

            elif isinstance(config, list):
                return cache([
                    self._eval((*config_key, i), (), trace)
                    for i in range(len(config))])
            elif isinstance(config, str) and config.startswith('@'):
                return cache(self._eval(
                    *_parse_key(config[1:], config_key[:-1])[:2], trace))
            else:
                return cache(config)

    def update_via_args(self, args: Sequence[Tuple[str, Any]]) -> None:
        for k, v in args:
            n_k, c_k = _parse_key(k, ())[:2]
            if (n_k, c_k) in self._cache:
                if (
                    isinstance(self._cache[(n_k, c_k)], bool)
                    and isinstance(v, str)
                ):
                    if not v.lower() in ("true", "false"):
                        raise ValueError(
                            f'bool should be true/false. Found {v}'
                        )
                    v = v.lower() == "true"
                self._cache[(n_k, c_k)] = type(self._cache[(n_k, c_k)])(v)
            else:
                self._cache[(n_k, c_k)] = v


def _parse_key(
        key: str, current_config_key: ConfigKey
) -> Tuple[ConfigKey, Optional[AttrKey], bool]:
    if key.startswith('!'):
        key = key[1:]
        escape = True
    else:
        escape = False

    if key.startswith('/'):
        key = key[1:]
        rel = False
    else:
        rel = True

    config_key_str = key.split('/')
    config_key_str[-1], *attr_key_list = config_key_str[-1].split('.')

    config_key = [_parse_k(k) for k in config_key_str]
    attr_key: Optional[AttrKey] = tuple(_parse_k(k) for k in attr_key_list)

    if escape:
        assert not attr_key
        attr_key = None

    if rel:
        config_key = list(current_config_key) + config_key

    i = 0
    while i < len(config_key):
        if config_key[i] in {'', '.'}:
            config_key.pop(i)
        elif config_key[i] == '..':
            assert i > 0
            config_key.pop(i)
            config_key.pop(i - 1)
            i -= 1
        else:
            i += 1

    return tuple(config_key), attr_key, rel


def _parse_k(k: str) -> Union[str, int]:
    try:
        return int(k)
    except ValueError:
        return k


def _dump_key(config_key: ConfigKey, attr_key: Optional[AttrKey]) -> str:
    config_key_str = '/' + '/'.join(str(k) for k in config_key)

    if attr_key:
        attr_key_str = '.'.join(str(k) for k in attr_key)
        return config_key_str + '.' + attr_key_str
    elif attr_key is None:
        return '!' + config_key_str
    else:
        return config_key_str


def _dump_trace(trace: DumpTrace) -> str:
    return ' -> '.join(
        _dump_key(config_key, attr_key)
        for config_key, attr_key in trace)


def _load(path: str, loader: Loader, trace: LoadTrace) -> ConfigType:
    path = os.path.normpath(path)
    circular = (path, ()) in trace
    trace = (*trace, (path, ()))
    if circular:
        raise RuntimeError(
            'Circular import',
            ' -> '.join('{} of {}'.format(_dump_key(config_key, None), path)
                        for path, config_key in trace))
    config = loader(path)
    return _expand_import(config, os.path.dirname(path), loader, trace)


def _expand_import(
        config: ConfigType,
        workdir: str,
        loader: Loader,
        trace: LoadTrace,
) -> ConfigType:
    path, config_key = trace[-1]
    if isinstance(config, dict):
        config = {k: _expand_import(v, workdir, loader,
                                    (*trace, (path, (*config_key, k))))
                  for k, v in config.items()}
        if 'import' in config:
            path = config['import']
            if not os.path.isabs(path):
                path = os.path.join(workdir, path)
            config_orig, config = config, _load(path, loader, trace)
            for k, v in config_orig.items():
                if k == 'import':
                    continue
                config_key, attr_key, rel = _parse_key(k, ())
                assert attr_key == ()
                assert rel

                c = config
                try:
                    for k in config_key[:-1]:
                        c = c[k]
                    c[config_key[-1]] = v
                except Exception as e:
                    e.args = e.args + (
                        '{} not in {}'.format(
                            _dump_key(config_key, attr_key), path),)
                    raise e

        return config
    elif isinstance(config, list):
        return [_expand_import(v, workdir, loader,
                               (*trace, (path, (*config_key, i))))
                for i, v in enumerate(config)]
    else:
        return config


def _json_loader(path: str) -> Any:
    with open(path) as f:
        return json.load(f)
