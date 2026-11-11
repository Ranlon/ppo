"""CPU tiny-model integration tests; no real model training or CUDA allocation."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import os

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast, TrainerState
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / 'full_runtime'), str(PROJECT / 'runtime'), str(PROJECT / 'tools')]
import project
from full_support import (make_profile, model_hashes, full_model_identity, attach_full_sft,
                          restore_full_weights, short_tmpdir)
from full_head import batched_forward, install_entropy_dispatch
from full_rollout import DenseNormPatch
from full_checkpoint import save_complete, load_complete
from select_full_gpu import select


def model():
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        pad_token_id=0, eos_token_id=1, attention_dropout=0.0, tie_word_embeddings=True))
    base.config.summary_dropout_prob = 0.0
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return AutoModelForCausalLMWithValueHead.from_pretrained(base)


def tokenizer():
    core = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token='2'))
    return PreTrainedTokenizerFast(tokenizer_object=core, pad_token='0', eos_token='1', unk_token='2')


def trainer(actor, ref, optimized):
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    t = PPOTrainer(PPOConfig(batch_size=3, mini_batch_size=3, gradient_accumulation_steps=1,
        ppo_epochs=4, learning_rate=1e-5, log_with=None, seed=42), model=actor,
        ref_model=ref, tokenizer=tokenizer(), optimizer=optimizer, lr_scheduler=scheduler)
    t.model_params = tuple(t.model_params)
    t.amp_context = __import__('contextlib').nullcontext()
    t.finetuning_args = types.SimpleNamespace(ppo_batch_python_gc=optimized)
    t._m9_head_config = dict(mode='bf16_masked', chunk=32, checkpoint='all', trunk='padded')
    if optimized:
        t.batched_forward_pass = types.MethodType(batched_forward, t)
    else:
        from llamafactory.train.ppo.trainer import CustomPPOTrainer
        t.batched_forward_pass = types.MethodType(CustomPPOTrainer.batched_forward_pass, t)
    from repair_policy import k3
    t._kl_penalty = types.MethodType(lambda self, a, b: k3(a, b), t)
    return t


class FullTuningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        install_entropy_dispatch()

    def test_profiles_are_full_and_preserve_common_ppo(self):
        a, b = make_profile('baseline'), make_profile('optimized')
        for profile in [a, b]:
            for stage in ['sft', 'ppo']:
                args = profile['common'][stage]
                self.assertEqual(args['finetuning_type'], 'full')
                self.assertIsInstance(args['learning_rate'], float)
                self.assertIsInstance(args['adam_epsilon'], float)
                self.assertFalse(any(k.startswith('lora_') for k in args))
        self.assertEqual(a['common']['ppo'], b['common']['ppo'])
        self.assertEqual(a['sft']['checkpoint_policy'], b['sft']['checkpoint_policy'])
        self.assertIsNone(a['sft']['odb'])
        self.assertEqual(b['sft']['odb']['gradient_accumulation_steps'], 1)
        self.assertEqual(a['common']['full_ppo_protection']['learning_rate'], 2e-7)
        self.assertTrue(a['common']['full_ppo_protection']['early_stopping'])
        self.assertEqual(a['common']['full_ppo_protection']['horizon'], 500)

    def test_full_ppo_overrides_legacy_lora_learning_rate_contract(self):
        profile=make_profile('baseline')
        self.assertEqual(profile['common']['ppo']['learning_rate'],2e-7)
        self.assertNotEqual(profile['common']['ppo']['learning_rate'],
                            profile['common']['trl_algorithm_contract']['learning_rate'])

    def test_project_tmpdir_supports_unix_managers_and_parallel_map(self):
        import multiprocessing
        import multiprocess
        from datasets import Dataset
        # Exercise the real socket creation that failed, including datasets.map.
        with patch.dict(os.environ, TMPDIR=str(short_tmpdir())), patch.object(tempfile, 'tempdir', None):
            for factory in [multiprocessing.Manager, multiprocess.Manager]:
                with factory() as manager:
                    shared=manager.dict()
                    shared['answer']=42
                    self.assertEqual(shared['answer'],42)
            dataset=Dataset.from_dict({'x':list(range(20))})
            mapped=dataset.map(lambda row: {'y': row['x']+1}, num_proc=4)
            self.assertEqual(mapped['y'],list(range(1,21)))

    def test_head_gradients_and_distinct_reference(self):
        torch.manual_seed(4)
        raw = model()
        ref = copy.deepcopy(raw).eval().requires_grad_(False)
        direct = copy.deepcopy(raw)
        t = trainer(raw, ref, True)
        queries = [torch.tensor([3, 4]), torch.tensor([5, 6]), torch.tensor([7, 8])]
        responses = [torch.tensor([9, 1]), torch.tensor([10, 1]), torch.tensor([11, 1])]
        inputs = dict(input_ids=torch.stack([torch.cat([q, r]) for q, r in zip(queries, responses)]),
                      attention_mask=torch.ones((3, 4), dtype=torch.long))
        ref_hash = model_hashes(ref)
        logp, _, values, mask = batched_forward(t, raw, queries, responses, inputs)
        logits, _, expected_values = direct(**inputs)
        expected_logp = logits[:, :-1].log_softmax(-1).gather(-1, inputs['input_ids'][:, 1:, None]).squeeze(-1)
        torch.testing.assert_close(logp[mask.bool()], expected_logp[mask.bool()])
        loss = ((logp + values) * mask).sum()
        expected_loss = ((expected_logp + expected_values[:, :-1]) * mask).sum()
        loss.backward(); expected_loss.backward()
        for (name, param), (other, expected) in zip(raw.named_parameters(), direct.named_parameters()):
            self.assertEqual(name, other)
            self.assertIsNotNone(param.grad, name)
            torch.testing.assert_close(param.grad, expected.grad, atol=2e-6, rtol=2e-4)
        t.optimizer.step()
        self.assertEqual(model_hashes(ref), ref_hash)
        self.assertNotEqual(model_hashes(raw), model_hashes(direct))
        self.assertIs(raw.pretrained_model.lm_head.weight, raw.pretrained_model.model.embed_tokens.weight)

    def test_real_ppo_updates_all_full_weights_in_both_paths(self):
        for optimized in [False, True]:
            with self.subTest(optimized=optimized):
                torch.manual_seed(12)
                actor = model()
                ref = copy.deepcopy(actor).eval().requires_grad_(False)
                before = model_hashes(actor)
                frozen = model_hashes(ref)
                t = trainer(actor, ref, optimized)
                calls = []
                original_step = t.optimizer.step
                def step(*args, **kwargs):
                    calls.append(1)
                    return original_step(*args, **kwargs)
                t.optimizer.step = step
                stats = t.step([torch.tensor([3, 4]), torch.tensor([5, 6, 7]), torch.tensor([8, 9])],
                               [torch.tensor([10, 11, 1]), torch.tensor([12, 1]), torch.tensor([13, 14, 1])],
                               [torch.tensor(0.), torch.tensor(1.), torch.tensor(1.)])
                self.assertEqual(len(calls), 4)
                self.assertTrue(math.isfinite(float(stats['ppo/loss/total'])))
                self.assertEqual(frozen, model_hashes(ref))
                after = model_hashes(actor)
                self.assertTrue(all(before[k] != after[k] for k in before), [k for k in before if before[k] == after[k]])
                self.assertFalse(t.is_peft_model)

    def test_bf16_full_head_keeps_tied_embedding_gradients(self):
        torch.manual_seed(25)
        actor=model()
        direct=copy.deepcopy(actor)
        ref=copy.deepcopy(actor).eval().requires_grad_(False)
        t=trainer(actor,ref,True)
        t.amp_context=torch.autocast('cpu',dtype=torch.bfloat16)
        queries=[torch.tensor([3,4]),torch.tensor([5,6]),torch.tensor([7,8])]
        responses=[torch.tensor([9,1]),torch.tensor([10,1]),torch.tensor([11,1])]
        inputs=dict(input_ids=torch.stack([torch.cat([q,r]) for q,r in zip(queries,responses)]),
                    attention_mask=torch.ones((3,4),dtype=torch.long))
        logp,_,values,mask=batched_forward(t,actor,queries,responses,inputs)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            logits,_,expected_values=direct(**inputs)
        expected_logp=logits[:,:-1].log_softmax(-1).gather(-1,inputs['input_ids'][:,1:,None]).squeeze(-1)
        torch.testing.assert_close(logp[mask.bool()],expected_logp[mask.bool()])
        ((logp+values)*mask).sum().backward()
        ((expected_logp+expected_values[:,:-1])*mask).sum().backward()
        for (name,p),(other,expected) in zip(actor.named_parameters(),direct.named_parameters()):
            self.assertEqual(name,other)
            torch.testing.assert_close(p.grad,expected.grad,atol=1e-5,rtol=1e-4,msg=name)

    def test_full_checkpoint_restores_parameters_adam_and_next_update(self):
        from compare_resume import compare
        torch.manual_seed(13)
        actor = model()
        ref = copy.deepcopy(actor).eval().requires_grad_(False)
        t = trainer(actor, ref, True)
        # Populate every Adam state using a real dense backward/update.
        ids = torch.tensor([[3, 4, 5, 1]])
        logits, _, values = actor(input_ids=ids)
        (logits.square().mean() + values.square().mean()).backward()
        t.optimizer.step(); t.optimizer.zero_grad(); t.lr_scheduler.step()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'runs/control'; output.mkdir(parents=True)
            frozen = model_hashes(ref)
            t.root=root; t.output=output; t.state=TrainerState(global_step=1)
            t.state_save_every=250; t.program_start=__import__('time').perf_counter()
            t.elapsed_before_resume=0.; t.config_sha='test'; t.sft_sha='sft'; t.reward_revision='reward'
            t.variant='compiled'; t.strict=False; t.sample_order=[0,1,2]
            t.rows=[dict(batch=1)]; t.rollouts=[dict(index=i) for i in range(3)]
            t.initial_trainable_sha=model_hashes(actor); t.zero_reward_streak=0; t.quality_stop_reason=None
            def check(): self.assertEqual(model_hashes(ref), frozen)
            t.check_frozen_reference=check
            save_complete(t, force=True)
            state_path=output/'state_checkpoints/step_000001/training_state.pt'
            expected_parameters = copy.deepcopy(actor.state_dict())
            expected_optimizer = copy.deepcopy(t.optimizer.state_dict())
            with torch.no_grad():
                for param in actor.parameters(): param.add_(1.)
            t.output=root/'runs/resumed'; t.output.mkdir()
            load_complete(t, state_path)
            compare(actor.state_dict(), expected_parameters, 'full weights')
            compare(t.optimizer.state_dict(), expected_optimizer, 'Adam')
            expected_actor=copy.deepcopy(actor)
            expected_adam=torch.optim.AdamW(expected_actor.parameters(),lr=1e-5)
            expected_adam.load_state_dict(copy.deepcopy(expected_optimizer))
            for m, opt in [(actor,t.optimizer),(expected_actor,expected_adam)]:
                z,_,v=m(input_ids=ids)
                (z.square().mean()+v.square().mean()).backward();opt.step();opt.zero_grad()
            compare(actor.state_dict(),expected_actor.state_dict(),'next update')

    def test_full_hf_save_load_and_reference_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full = root/'sft/model'
            base = model().pretrained_model
            base.save_pretrained(full)
            identity = full_model_identity(full)
            (root/'REQUESTED_CONFIG.json').write_text(json.dumps(dict(requested_lf_arguments={})))
            attach_full_sft(root, full)
            request = json.loads((root/'REQUESTED_CONFIG.json').read_text())
            self.assertEqual(request['sft_model_sha256'],identity['sha256'])
            self.assertEqual(request['requested_lf_arguments']['ref_model'],str(full.resolve()))
            loaded=Qwen3ForCausalLM.from_pretrained(full)
            self.assertEqual(model_hashes(base),model_hashes(loaded))
            # Simulate the native load path that would silently round SFT values.
            loaded.to(torch.bfloat16).to(torch.float32)
            self.assertNotEqual(model_hashes(base),model_hashes(loaded))
            restore_full_weights(loaded,full)
            self.assertEqual(model_hashes(base),model_hashes(loaded))
            (root/'runs/old').mkdir(parents=True)
            with torch.no_grad(): base.model.layers[0].self_attn.q_proj.weight.add_(1.)
            base.save_pretrained(full)
            with self.assertRaises(ValueError):attach_full_sft(root,full)

    def test_ppo_hf_checkpoint_preserves_full_model_and_value_head(self):
        from llamafactory.train.callbacks import fix_valuehead_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            actor=model()
            actor.save_pretrained(folder, safe_serialization=True)
            fix_valuehead_checkpoint(actor,folder,True)
            loaded=Qwen3ForCausalLM.from_pretrained(folder)
            self.assertEqual(model_hashes(actor.pretrained_model),model_hashes(loaded))
            self.assertTrue((Path(folder)/'value_head.safetensors').exists())

    def test_tied_embedding_save_state_deduplicates_shared_storage(self):
        actor=model()
        raw=actor.state_dict()
        self.assertTrue(raw['lm_head.weight'].data_ptr()==raw['model.embed_tokens.weight'].data_ptr())
        state=dict(raw)
        for head_name, embedding_name in [('lm_head.weight','model.embed_tokens.weight')]:
            if state[head_name].data_ptr()==state[embedding_name].data_ptr():
                del state[head_name]
        with tempfile.TemporaryDirectory() as folder:
            from safetensors.torch import save_file
            save_file({k:v.detach().cpu().contiguous() for k,v in state.items()},
                      str(Path(folder)/'model.safetensors'))
            self.assertTrue((Path(folder)/'model.safetensors').stat().st_size > 0)

    def test_prepare_local_06b_without_training(self):
        from full_cli import prepare
        from full_support import verify_full, read_jsonl
        task_home=PROJECT.parents[1]
        base=task_home/'models/Qwen3-0.6B-Base'
        train=task_home/'datasets/gsm8k/train.jsonl'
        if not base.exists() or not train.exists():self.skipTest('Local model/data not present')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'prepared'
            args=types.SimpleNamespace(profile='optimized',odb_token_budget=1024,odb_buffer_size=64,
                sft_lr=2e-5,ppo_lr=1e-6,model=str(base),run_dir=str(root),train=str(train),
                test=str(train.with_name('test.jsonl')))
            prepare(args)
            verify_full(root)
            self.assertEqual(len(read_jsonl(root/'data/train.jsonl')),7473)
            self.assertNotIn(chr(0x2028),(root/'data/train.jsonl').read_text())

    def test_full_ppo_launcher_accepts_full_sft_and_cleans_reward_server(self):
        from full_cli import run_ppo
        from full_support import build_manifest
        from project import sha, write
        from urllib.request import Request, urlopen
        # Run the actual launcher/scorer boundary; replace only the GPU trainer.
        # No adapter file or adapter SHA is present anywhere in this fixture.
        for variant in ['baseline', 'optimized']:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                full=root/'sft/model'
                model().pretrained_model.save_pretrained(full)
                identity=full_model_identity(full)
                data=root/'data';data.mkdir()
                train=data/'train.jsonl'
                train.write_text(''.join(json.dumps(dict(question=f'Q{i}',answer='#### 1'))+'\n'
                                         for i in range(7473)))
                write(root/'BASE_MODEL.json',identity)
                write(root/'INPUT_IDENTITY.json',dict(model=identity['files'],
                    data={'train':dict(sha256=sha(train),examples=7473)},prepared_data={}))
                write(root/'PROFILE.json',make_profile(variant))
                write(root/'REQUESTED_CONFIG.json',dict(requested_lf_arguments={}))
                attach_full_sft(root,full)
                write(root/'FULL_RUNTIME_BUILD.json',build_manifest())
                write(root/'REWARD_SPEC.json',dict(train_sha256=sha(train),
                    scorer_source_sha256=sha(PROJECT/'runtime/numeric_reward.py'),
                    model='test numeric rule',revision='test'))
                args=types.SimpleNamespace(name='full',sft=None,limit=1,save_every=250,
                                           resume_state=None,strict=False,stress=False)
                def gpu_launch(run_root,script,*arguments):
                    self.assertEqual(script,'full_train_ppo.py')
                    self.assertEqual(arguments[arguments.index('--variant')+1],
                                     'native' if variant=='baseline' else 'compiled')
                    url=arguments[arguments.index('--reward-url')+1]
                    payload=json.dumps(dict(prompts=['Q0'],responses=['#### 1'],indices=[0],run='full')).encode()
                    req=Request(url,data=payload,headers={'Content-Type':'application/json'})
                    with urlopen(req,timeout=5) as response:self.assertEqual(json.load(response)['scores'],[1.0])
                with patch('full_cli.launch',side_effect=gpu_launch) as launch:
                    run_ppo(root,args)
                    self.assertEqual(launch.call_count,1)
                ready=json.loads((root/'REWARD_SERVER_full.json').read_text())
                with self.assertRaises(ProcessLookupError):os.kill(ready['pid'],0)

    def test_norm_restores_and_gpu_selection_respects_visibility(self):
        base=model().pretrained_model
        hashes=model_hashes(base)
        with torch.no_grad():expected=base(torch.tensor([[3,4,1]])).logits
        patch=DenseNormPatch(base)
        with self.assertRaises(RuntimeError):
            with patch.active():
                with torch.no_grad():actual=base(torch.tensor([[3,4,1]])).logits
                torch.testing.assert_close(actual,expected)
                raise RuntimeError('probe restoration')
        self.assertEqual(model_hashes(base),hashes)
        self.assertTrue(all('forward' not in m.__dict__ for m,_,_,_ in patch.entries))
        report='0, GPU-a, A800, 8.0, 28000, 0\n4, GPU-b, A800, 8.0, 41718, 0\n7, GPU-c, A800, 8.0, 31000, 0\n'
        self.assertEqual(select(report)[2],'GPU-b')
        self.assertEqual(select(report,visible='0,7')[2],'GPU-c')
        self.assertEqual(select(report,requested='0')[2],'GPU-a')
        with self.assertRaises(ValueError):select(report,visible='')


if __name__ == '__main__':
    unittest.main()
