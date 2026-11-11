"""Observe native and compiler cleanup separately; never suppress compiler calls."""
import collections
import gc
import sys
import time
from cleanup_policy import CleanupPolicy


class RuntimeCleanupAudit(CleanupPolicy):
    def __init__(self,policy,torch,native_python_gc,expected_cache_calls=11):
        assert policy=='native'
        self.native_python_gc=native_python_gc
        self.expected_cache_calls=expected_cache_calls
        self.sites=collections.Counter()
        super().__init__(policy,torch)

    def start(self):
        super().start();self.sites.clear()

    def source(self,operation):
        frame=sys._getframe(2)
        filename=frame.f_code.co_filename.replace('\\','/')
        function=frame.f_code.co_name
        category='native' if '/llamafactory/' in filename or '/trl/' in filename else 'torch_compiler' if '/torch/' in filename else 'other'
        self.counters[category+'_'+operation]+=1
        self.sites[filename+':'+function+':'+operation]+=1

    def collect(self,*args,**kwargs):
        if not self.active:return self.original_gc(*args,**kwargs)
        self.source('gc');self.counters['gc_requested']+=1
        return self.execute('gc',self.original_gc,*args,**kwargs)

    def empty_cache(self,*args,**kwargs):
        if not self.active:return self.original_cache(*args,**kwargs)
        self.source('cache');self.counters['cache_requested']+=1
        return self.execute('cache',self.original_cache,*args,**kwargs)

    def finish(self):
        assert self.active
        try:
            result=dict(counts=dict(self.counters),cpu_wall_seconds=dict(self.seconds),automatic_gc_enabled=gc.isenabled(),call_sites=dict(self.sites))
            assert self.counters['native_gc']==(1 if self.native_python_gc else 16),result
            assert self.counters['native_cache']==self.expected_cache_calls,result
            assert self.counters['gc_executed']==self.counters['gc_requested']
            assert self.counters['cache_executed']==self.counters['cache_requested']
            assert not self.counters['gc_suppressed'] and not self.counters['cache_suppressed']
            return result
        finally:self.active=False
