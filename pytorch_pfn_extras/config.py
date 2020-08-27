import json
import os
import reprlib


def customize_type(**default_kwargs):
    def deco(type_):
        type_._custom_default_kwargs = default_kwargs
        return type_
    return deco


class Config(object):

    def __init__(self, config, types=None):
        self._cache = {((), None): config}
        self._types = types or {}

    def __getitem__(self, key):
        return self._eval(*_parse_key(key, None)[:2], ())

    @classmethod
    def load_path(cls, path, *, loader=None, types=None):
        if loader is None:
            loader = _json_loader
        return cls(_load(path, loader, ()), types)

    def _eval(self, config_key, attr_key, trace):
        if (config_key, attr_key) in self._cache:
            return self._cache[(config_key, attr_key)]

        circular = (config_key, attr_key) in trace
        trace = (*trace, (config_key, attr_key))
        if circular:
            raise RuntimeError(
                'Circular dependency',
                _dump_trace(trace))

        def cache(value):
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


def _parse_key(key, current_config_key):
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

    config_key = key.split('/')
    config_key[-1], *attr_key = config_key[-1].split('.')

    config_key = [_parse_k(k) for k in config_key]
    attr_key = tuple(_parse_k(k) for k in attr_key)

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


def _parse_k(k):
    try:
        return int(k)
    except ValueError:
        return k


def _dump_key(config_key, attr_key):
    config_key = '/' + '/'.join(str(k) for k in config_key)

    if attr_key:
        attr_key = '.'.join(str(k) for k in attr_key)
        return config_key + '.' + attr_key
    elif attr_key is None:
        return '!' + config_key
    else:
        return config_key


def _dump_trace(trace):
    return ' -> '.join(
        _dump_key(config_key, attr_key)
        for config_key, attr_key in trace)


def _load(path, loader, trace):
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


def _expand_import(config, workdir, loader, trace):
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


def _json_loader(path):
    with open(path) as f:
        return json.load(f)
