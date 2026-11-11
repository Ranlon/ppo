"""Static-cache compiled rollout for mutable full model weights; no LoRA merge."""
import contextlib
import copy
import random
import time
import types
import numpy as np
import torch
from transformers import CompileConfig, StaticCache
from project import runtime_options


class DenseNormPatch:
    def __init__(self, model):
        from staged_norm import staged
        self.entries = []
        for module in model.modules():
            if type(module).__name__ != 'Qwen3RMSNorm':
                continue
            def forward(this, x):
                return staged(x, this.weight, float(this.variance_epsilon))
            self.entries.append((module, types.MethodType(forward, module),
                                 'forward' in module.__dict__, module.__dict__.get('forward')))
        assert self.entries

    @contextlib.contextmanager
    def active(self):
        try:
            for module, forward, _, _ in self.entries:
                module.forward = forward
            yield
        finally:
            for module, _, had, old in self.entries:
                if had:
                    module.forward = old
                else:
                    del module.forward


class DenseCompiledRollout:
    def __init__(self, model, batch_size, max_cache_len):
        import torch._inductor.config as config
        options = runtime_options()
        assert options['rollout'] == 'compiled_dense' and batch_size == 3
        assert not hasattr(model, 'peft_config')
        config.compile_threads = 2
        # Full actor weights change every batch. Never freeze/copy constants.
        config.freezing = False
        config.triton.cudagraphs = True
        config.emulate_precision_casts = True
        config.force_same_precision = True
        self.model = model
        self.max_cache_len = options['static_cache_length']
        self.norm = DenseNormPatch(model)
        self.cache = StaticCache(config=model.config, max_cache_len=self.max_cache_len)
        weight = model.get_input_embeddings().weight
        sample = torch.empty((batch_size, model.config.num_key_value_heads, 1, model.config.head_dim),
                             dtype=torch.bfloat16, device=weight.device)
        for layer in self.cache.layers:
            layer.lazy_initialization(sample)
        self.compile_config = CompileConfig(backend='inductor', fullgraph=True, dynamic=False,
                                            mode='max-autotune')
        self.identities = {name: (id(p), p.data_ptr()) for name, p in model.named_parameters()}
        self.calls = 0
        self.initialization_seconds = None

    def check_identity(self):
        assert self.identities == {name: (id(p), p.data_ptr()) for name, p in self.model.named_parameters()}
        assert self.model.get_input_embeddings().weight is self.model.get_output_embeddings().weight

    def call(self, **kwargs):
        self.check_identity()
        maximum = kwargs.get('max_new_tokens', kwargs.get('generation_config').max_new_tokens
                             if kwargs.get('generation_config') else 512)
        assert kwargs['input_ids'].shape[1] + maximum <= self.max_cache_len
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16), self.norm.active():
                self.cache.reset()
                torch.compiler.cudagraph_mark_step_begin()
                result = self.model.generate(**kwargs, past_key_values=self.cache,
                    disable_compile=False, compile_config=self.compile_config)
                # Do not retain output aliases into a CUDA graph across training.
                result = result.clone()
        finally:
            self.check_identity()
        self.calls += 1
        return result

    def generate(self, **kwargs):
        if self.initialization_seconds is None:
            started = time.perf_counter()
            rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                   torch.cuda.get_rng_state_all())
            warm = dict(kwargs)
            if 'generation_config' in warm:
                warm['generation_config'] = copy.deepcopy(warm['generation_config'])
            warm.update(max_new_tokens=4, min_new_tokens=4, do_sample=False, top_k=None)
            try:
                self.call(**warm)
                torch.cuda.synchronize()
            finally:
                random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
                torch.cuda.set_rng_state_all(rng[3])
            self.initialization_seconds = time.perf_counter() - started
        return self.call(**kwargs)

    def statistics(self):
        return dict(mode='compiled_full_dense', calls=self.calls, cache_length=self.max_cache_len,
                    initialization_seconds=self.initialization_seconds, backend='inductor',
                    norm='staged_native_mean', freezing=False, lora_merge=False,
                    backup_bytes=0, independent_head_bytes=0)
