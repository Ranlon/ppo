"""Process-local explicit cleanup cadence; automatic Python GC stays enabled."""
import collections
import gc
import time


class CleanupPolicy:
    def __init__(self, policy, torch):
        assert policy == 'native'
        self.policy = policy
        self.torch = torch
        self.original_gc = gc.collect
        self.original_cache = torch.cuda.empty_cache
        self.active = False
        self.counters = collections.Counter()
        self.seconds = collections.Counter()
        gc.collect = self.collect
        torch.cuda.empty_cache = self.empty_cache

    def start(self):
        assert not self.active
        self.counters.clear()
        self.seconds.clear()
        self.active = True

    def execute(self, name, function, *args, **kwargs):
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            self.seconds[name] += time.perf_counter() - started
            self.counters[name + '_executed'] += 1

    def collect(self, *args, **kwargs):
        if not self.active:
            return self.original_gc(*args, **kwargs)
        self.counters['gc_requested'] += 1
        return self.execute('gc', self.original_gc, *args, **kwargs)

    def empty_cache(self, *args, **kwargs):
        if not self.active:
            return self.original_cache(*args, **kwargs)
        self.counters['cache_requested'] += 1
        return self.execute('cache', self.original_cache, *args, **kwargs)

    def finish(self):
        assert self.active
        # Run only after native PPO returns, so its temporary tensors can be released.
        try:
            result = dict(counts=dict(self.counters), cpu_wall_seconds=dict(self.seconds),
                          automatic_gc_enabled=gc.isenabled())
            assert self.counters['gc_requested'] == 16, result
            assert self.counters['cache_requested'] == 11, result
            assert (self.counters['gc_executed'], self.counters['cache_executed']) == (16, 11)
            return result
        finally:
            self.active = False

    def restore(self):
        self.active = False
        gc.collect = self.original_gc
        self.torch.cuda.empty_cache = self.original_cache
