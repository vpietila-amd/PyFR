import time

import numpy as np


class Timer:
    """Times device work submitted to a backend's queue."""

    def __init__(self, backend):
        self.backend = backend

        if backend.name == 'hip':
            self._stream = backend._stream
            self._evts = backend.hip.create_event(), backend.hip.create_event()

    def _time_once(self, fn):
        if self.backend.name == 'hip':
            start, stop = self._evts
            start.record(self._stream)
            fn()
            stop.record(self._stream)
            stop.synchronize()

            return stop.elapsed_time(start)
        else:
            self.backend.wait()
            t0 = time.perf_counter()
            fn()
            self.backend.wait()

            return time.perf_counter() - t0

    def __call__(self, fn, reps=20, warmup=2):
        for _ in range(warmup):
            self._time_once(fn)

        return np.array([self._time_once(fn) for _ in range(reps)])


def stats(times):
    # The median is authoritative; idle clock ramps inflate the mean
    return {
        'reps': len(times),
        'median_s': float(np.median(times)),
        'min_s': float(np.min(times)),
        'p90_s': float(np.percentile(times, 90)),
        'mean_s': float(np.mean(times)),
        'std_s': float(np.std(times)),
    }
