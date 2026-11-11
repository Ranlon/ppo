"""Compiled rollout with full temporary merge for larger GPUs.

Merging is inference-only and numerically approximate in BF16. Restore is exact.
The original tied embedding/head Parameter and every LoRA Parameter are retained.
"""
import contextlib
import copy
import random
import time
import types
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CompileConfig, StaticCache
from project import runtime_options


@torch.library.custom_op('ppo4portable::linear', mutates_args=())
def native_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.linear(x, w)

@native_linear.register_fake
def _(x, w):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


def vanilla_layers(model):
    from peft.tuners.lora.layer import Linear
    result = []
    for name, module in model.named_modules():
        if not isinstance(module, Linear):
            continue
        assert list(module.active_adapters) == ['default'] and not module.merged
        assert not module.disable_adapters
        assert getattr(module.lora_dropout['default'],'p',0) == 0
        assert not module.use_dora.get('default', False)
        assert not getattr(module, 'lora_variant', {}).get('default')
        assert module.base_layer.bias is None
        assert not module.base_layer.weight.requires_grad
        assert not module.fan_in_fan_out
        result.append((name, module))
    return result


def restore_forward(module, had, old):
    if had:
        module.forward = old
    else:
        del module.forward


class ExactMerge:
    """Dense BF16 backups; GPU or pinned CPU. No subtract-delta restoration."""
    def __init__(self, model, backup='gpu', chunk_rows=1024):
        assert backup in ('gpu', 'cpu') and chunk_rows > 0
        self.model, self.backup, self.chunk_rows = model, backup, chunk_rows
        self.embedding = model.get_input_embeddings().weight
        self.head = model.get_output_embeddings()
        assert self.head.base_layer.weight is self.embedding
        self.head_original = self.head.base_layer.weight
        self.entries = []
        self.modified = []
        self.active_now = False
        self.merge_seconds = self.restore_seconds = 0.
        self.calls = 0
        for name, module in vanilla_layers(model):
            weight = module.base_layer.weight
            is_head = module is self.head
            if is_head:
                saved = None
            else:
                device = weight.device if backup == 'gpu' else torch.device('cpu')
                saved = torch.empty_like(weight, device=device,
                    pin_memory=(backup == 'cpu' and weight.device.type == 'cuda'))
                with torch.no_grad(): saved.copy_(weight)
            self.entries.append((name, module, weight, saved, is_head,
                                 'forward' in module.__dict__, module.__dict__.get('forward')))
        assert sum(entry[4] for entry in self.entries) == 1
        self.head_work = torch.nn.Parameter(torch.empty_like(self.embedding), requires_grad=False)
        if self.embedding.device.type == 'cuda':
            torch._dynamo.mark_static_address(self.head_work)
            for _, _, _, saved, _, _, _ in self.entries:
                if saved is not None and saved.device.type == 'cuda': torch._dynamo.mark_static_address(saved)
        self.backup_bytes = sum(s.numel()*s.element_size() for _,_,_,s,_,_,_ in self.entries if s is not None)
        self.head_work_bytes = self.head_work.numel()*self.head_work.element_size()
        self.identities = {id(p): (p, p.data_ptr()) for p in model.parameters()}

    def synchronize(self):
        if self.embedding.device.type == 'cuda': torch.cuda.synchronize(self.embedding.device)

    @torch.no_grad()
    def add_delta(self, destination, module):
        # Current FP32 factors, recomputed every generation after PPO updates.
        a = module.lora_A['default'].weight
        b = module.lora_B['default'].weight
        for start in range(0, b.shape[0], self.chunk_rows):
            end = min(start+self.chunk_rows, b.shape[0])
            delta = (b[start:end].float() @ a.float()) * module.scaling['default']
            destination[start:end].add_(delta)

    @torch.no_grad()
    def verify_exact(self):
        assert self.head.base_layer.weight is self.head_original is self.embedding
        assert self.model.get_input_embeddings().weight is self.embedding
        for _, module, weight, saved, is_head, _, _ in self.entries:
            assert module.base_layer.weight is weight
            assert not module.merged
            if not is_head:
                actual = weight if saved.device == weight.device else weight.cpu()
                assert torch.equal(actual, saved), 'Base restoration mismatch'
        for parameter, pointer in self.identities.values():
            assert parameter.data_ptr() == pointer, 'Parameter identity/storage changed'
        assert set(self.identities) == {id(p) for p in self.model.parameters()}
        assert not self.active_now and not getattr(self.model, '_ppo4_merge_active', False)

    @contextlib.contextmanager
    def active(self):
        assert not self.active_now and not getattr(self.model, '_ppo4_merge_active', False)
        assert list(self.model.active_adapters) == ['default']
        self.active_now = self.model._ppo4_merge_active = True
        self.modified = []
        started = time.perf_counter()
        try:
            with torch.no_grad():
                for entry in self.entries:
                    _, module, weight, saved, is_head, had, old = entry
                    assert not module.merged
                    self.modified.append(entry)  # Restore even if the next write fails.
                    if is_head:
                        self.head_work.copy_(self.head_original)
                        self.add_delta(self.head_work, module)
                        module.base_layer.weight = self.head_work
                    else:
                        self.add_delta(weight, module)
                    module.merged_adapters.append('default')
                    def forward(this, x, *args, **kwargs):
                        assert not args and not kwargs
                        return native_linear(x.to(this.base_layer.weight.dtype), this.base_layer.weight)
                    module.forward = types.MethodType(forward, module)
            self.synchronize(); self.merge_seconds += time.perf_counter()-started
            yield
        finally:
            started = time.perf_counter()
            with torch.no_grad():
                for _, module, weight, saved, is_head, had, old in reversed(self.modified):
                    if is_head: module.base_layer.weight = self.head_original
                    else: weight.copy_(saved, non_blocking=False)
                    module.merged_adapters.clear()
                    restore_forward(module, had, old)
            self.active_now = self.model._ppo4_merge_active = False
            self.synchronize()
            self.verify_exact()
            self.restore_seconds += time.perf_counter()-started
            self.calls += 1

    def statistics(self):
        return dict(mode='full_temporary_merge', backup_device=self.backup,
                    backup_bytes=self.backup_bytes, independent_head_bytes=self.head_work_bytes,
                    generation_calls=self.calls, merge_seconds=self.merge_seconds,
                    restore_and_verification_seconds=self.restore_seconds,
                    exact_restore=True, bf16_forward_equivalence=False)


def install_torch26_scalar_guard():
    # Torch 2.6 CPU-int8 matcher must reject scalar LoRA multipliers safely.
    from torch._inductor import compile_fx
    from torch._inductor.fx_passes import quantization
    native = quantization._is_valid_woq_optimization_pattern
    if getattr(native, '_ppo4_guard', False):return
    def factory():
        check_native = native()
        def check(match):
            if not all(isinstance(match.kwargs.get(k),torch.fx.Node) for k in ('x','weight','scales')):return False
            return check_native(match)
        return check
    factory._ppo4_guard=True
    quantization._is_valid_woq_optimization_pattern=factory


class NativeRolloutAcceleration:
    def __init__(self, model, batch_size, max_cache_len):
        import torch._inductor.config as config
        from staged_norm import NormPatch
        options=runtime_options()
        assert options['rollout']=='merged_compiled'
        assert torch.__version__.startswith('2.6.0')
        config.compile_threads=2; config.freezing=False
        config.triton.cudagraphs=True
        config.emulate_precision_casts=True; config.force_same_precision=True
        install_torch26_scalar_guard()
        assert model.config.num_hidden_layers==36 and batch_size==3
        assert len(vanilla_layers(model))==253
        self.model=model;self.batch_size=batch_size
        self.max_cache_len=int(options['static_cache_length'])
        assert self.max_cache_len==768
        self.original_weight=model.get_input_embeddings().weight
        self.original_pointer=self.original_weight.data_ptr()
        self.patch=ExactMerge(model,options['merge_backup'],options['merge_chunk_rows'])
        self.norm=NormPatch(model,'staged')
        self.cache=StaticCache(config=model.config,max_cache_len=self.max_cache_len)
        with torch.no_grad():
            sample=torch.empty((batch_size,model.config.num_key_value_heads,1,model.config.head_dim),
                               dtype=torch.bfloat16,device=self.original_weight.device)
            for layer in self.cache.layers:layer.lazy_initialization(sample)
        self.compile_config=CompileConfig(backend='inductor',fullgraph=True,dynamic=False,mode='max-autotune')
        self.prepared=False;self.initialization_seconds=None;self.calls=0
        self.check_identity()

    def check_identity(self):
        assert self.model.get_input_embeddings().weight is self.original_weight
        assert self.model.get_output_embeddings().base_layer.weight is self.original_weight
        assert self.original_weight.data_ptr()==self.original_pointer
        assert list(self.model.active_adapters)==['default']
        assert not any(m.merged for _,m in vanilla_layers(self.model))
        assert not getattr(self.model,'_ppo4_merge_active',False)

    @contextlib.contextmanager
    def suspend_hooks(self):
        embedding=self.model.get_input_embeddings();original=embedding._forward_hooks
        filtered=original.copy()
        for key,hook in original.items():
            if getattr(hook,'__name__','') in ('make_inputs_require_grad','make_inputs_require_grads'):del filtered[key]
        try:
            embedding._forward_hooks=filtered
            yield
        finally:embedding._forward_hooks=original

    def call(self, **kwargs):
        self.check_identity()
        assert kwargs['input_ids'].shape[0]==self.batch_size
        maximum=kwargs.get('max_new_tokens',getattr(kwargs.get('generation_config'),'max_new_tokens',512))
        assert kwargs['input_ids'].shape[1]+maximum<=self.max_cache_len
        try:
            with self.suspend_hooks(),torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),self.norm.active(),self.patch.active():
                torch.compiler.cudagraph_mark_step_begin();self.cache.reset()
                output=self.model.generate(**kwargs,past_key_values=self.cache,disable_compile=False,compile_config=self.compile_config)
        finally:self.check_identity()
        self.calls+=1
        return output

    def generate(self, **kwargs):
        if not self.prepared:
            started=time.perf_counter()
            rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
            warm=dict(kwargs)
            if 'generation_config' in warm:warm['generation_config']=copy.deepcopy(warm['generation_config'])
            warm.update(max_new_tokens=4,min_new_tokens=4,do_sample=False)
            try:
                self.call(**warm);torch.cuda.synchronize()
            finally:
                random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])
            self.initialization_seconds=time.perf_counter()-started;self.prepared=True
        return self.call(**kwargs)

    def statistics(self):
        return dict(calls=self.calls,initialization_seconds=self.initialization_seconds,
                    cache_length=self.max_cache_len,norm='staged_native_mean',backend='inductor',
                    **self.patch.statistics())
