import os
import json
import io
import yaml

from pytorch_pfn_extras import reporting
from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training import trigger as trigger_module
from pfcv.profiler.time_summary import time_summary


def dump(stats, file):
    # TODO use writer
    s = io.StringIO()
    yaml.dump([stats], s)
    if not os.path.exists(file):
        with open(file, mode="w") as f:
            f.write(s.getvalue() + "\n")
    else:
        # append
        with open(file, mode="a") as f:
            f.write(s.getvalue() + "\n")


class ProfileReport(extension.Extension):
    def __init__(
        self,
        store_keys=None,
        report_keys=None,
        trigger=(1, "epoch"),
        filename=None,
        **kwargs,
    ):
        if store_keys is None:
            self._store_keys = store_keys
        else:
            self._store_keys = list(store_keys) + [
                key + ".std" for key in store_keys
            ]
        self._report_keys = report_keys
        self._trigger = trigger_module.get_trigger(trigger)
        self._log = []

        log_name = kwargs.get("log_name", "log")
        if filename is None:
            filename = log_name
        del log_name  # avoid accidental use
        self._log_name = filename

    def __call__(self, manager):
        if manager.is_before_training or self._trigger(manager):
            with time_summary.summary(clear=True) as s:
                stats = s.make_statistics()
            # report
            if self._report_keys is not None:
                reports = {
                    f"time.{k}": v
                    for k, v in stats.items()
                    if k in self._report_keys
                }
                reporting.report(reports)

            # output the result
            if self._store_keys is not None:
                stats = {
                    k: v for k, v in stats.items() if k in self._store_keys
                }
            stats_cpu = {}
            for name, value in stats.items():
                stats_cpu[name] = float(value)  # copy to CPU

            stats_cpu["epoch"] = manager.epoch
            stats_cpu["iteration"] = manager.iteration
            stats_cpu["elapsed_time"] = manager.elapsed_time

            # Recreate dict to fix order of logs
            out = {}
            keys = list(stats_cpu.keys())
            keys.sort()
            for key in keys:
                out[key] = stats_cpu[key]

            self._log.append(out)

            # write to the log file
            if self._log_name is not None:
                log_name = self._log_name.format(**out)
                dump(out, os.path.join(manager.out, log_name))

    def state_dict(self):
        state = {}
        if hasattr(self._trigger, "state_dict"):
            state["_trigger"] = self._trigger.state_dict()
        state["_log"] = json.dumps(self._log)
        return state

    def load_state_dict(self, to_load):
        if hasattr(self._trigger, "load_state_dict"):
            self._trigger.load_state_dict(to_load["_trigger"])
        self._log = json.loads(to_load["_log"])
