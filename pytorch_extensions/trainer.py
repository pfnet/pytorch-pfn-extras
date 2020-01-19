import collections
import os
import sys
import time
import traceback

import six

from pytorch_extensions import extension as extension_module
from pytorch_extensions import trigger as trigger_module
from pytorch_extensions import reporter as reporter_module


# Select the best-resolution timer function
try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time


class _ExtensionEntry(object):

    def __init__(self, extension, priority, trigger, call_before_training):
        self.extension = extension
        self.trigger = trigger
        self.priority = priority
        self.call_before_training = call_before_training

    def state_dict(self):
        state = {}
        if hasattr(self.extension, 'state_dict'):
            state['extension'] = self.extension.state_dict()
        if hasattr(self.trigger, 'state_dict'):
            state['trigger'] = self.trigger.state_dict()
        return state

    def load_state_dict(self, to_load):
        if 'extension' in to_load:
            self.extension.load_state_dict(to_load['extension'])
        if 'trigger' in to_load:
            self.trigger.load_state_dict(to_load['trigger'])
