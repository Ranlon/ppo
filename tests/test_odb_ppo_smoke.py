"""CPU checks for paired short-test isolation, metrics and real prepare compatibility."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'tools'))
from odb_ppo_smoke import probe_indices, choose_gpu, snapshot_source, sha
from odb_ppo_summary import phase_stats, paired_accuracy, summarize


class ODBPPOSmokeTests(unittest.TestCase):
    def test_fixed_probes_exclude_ppo_prefix_and_gpu_respects_visibility(self):
        indices=probe_indices(100,list(range(40)),12)
        self.assertEqual(indices,probe_indices(100,list(range(40)),12))
        self.assertFalse(set(indices)&set(range(40)))
        with self.assertRaises(ValueError):probe_indices(10,list(range(10)),1)
        report='1, GPU-a, A800, 8.0, 33000, 0\n4, GPU-b, A800, 8.0, 41000, 100\n7, GPU-c, A800, 8.0, 31000, 0\n'
        self.assertEqual(choose_gpu(report)[3],'GPU-a')
        self.assertEqual(choose_gpu(report,visible='7')[3],'GPU-c')
        with self.assertRaises(ValueError):choose_gpu(report,requested='4')

    def test_snapshot_fixes_only_compatibility_not_ppo_or_live_source(self):
        before=sha(PROJECT/'runtime/cli.py')
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source'
            snapshot_source(source)
            self.assertEqual(sha(PROJECT/'runtime/train_ppo.py'),sha(source/'runtime/train_ppo.py'))
            self.assertEqual(sha(PROJECT/'runtime/repair_policy.py'),sha(source/'runtime/repair_policy.py'))
            self.assertEqual(before,sha(PROJECT/'runtime/cli.py'))
            text=(source/'runtime/cli.py').read_text()
            self.assertIn("common=json.loads((PROJECT/'configs/common.yaml')",text)
            self.assertIn("json.dumps(r,ensure_ascii=True)+'\\n' for r in rows",text)
            self.assertFalse(json.loads((source/'SNAPSHOT_PATCHES.json').read_text())['algorithm_changes'])

    def test_actual_prepare_only_checks_both_real_sft_origins_and_reward_data(self):
        if importlib.util.find_spec('transformers') is None:
            self.skipTest('Use llm200 for dependency-backed preparation')
        pair=PROJECT/'outputs/sft_odb_compare_20261105_175703'
        if not (pair/'odb/sft/SUCCESS.json').exists():self.skipTest('Local SFT checkpoints not present')
        with tempfile.TemporaryDirectory(dir=PROJECT/'outputs') as tmp:
            output=Path(tmp)/'experiment'
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='',DISABLE_VERSION_CHECK='1',OMP_NUM_THREADS='1')
            run=subprocess.run([sys.executable,str(PROJECT/'tools/odb_ppo_smoke.py'),
                '--prepare-only','--output',str(output)],env=env,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+'\n'+run.stderr)
            exp=json.loads((output/'EXPERIMENT.json').read_text())
            plan=json.loads((output/'PROBE_PLAN.json').read_text())
            self.assertIsNone(exp['gpu'])
            self.assertEqual(len(plan['indices']),32)
            for name in ['baseline','odb']:
                root=output/name
                request=json.loads((root/'REQUESTED_CONFIG.json').read_text())
                self.assertEqual(request['sft_adapter_sha256'],exp['source_checkpoints'][name]['adapter_sha256'])
                self.assertIsInstance(request['requested_lf_arguments']['learning_rate'],float)
                self.assertIsInstance(request['requested_lf_arguments']['adam_epsilon'],float)
                self.assertFalse((root/'runs').exists())
                self.assertNotIn(chr(0x2028),(root/'data/train.jsonl').read_text())
            check=('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
                   'from numeric_reward import ExactNumericReward;'
                   'r=ExactNumericReward(Path(sys.argv[2]));assert len(r.rows)==7473')
            reward=subprocess.run([sys.executable,'-c',check,str(output/'source/runtime'),str(output/'odb')],
                                   env=env,capture_output=True,text=True)
            self.assertEqual(reward.returncode,0,reward.stderr)

    def test_summary_uses_common_prefix_and_before_after_quality(self):
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value))
        def lines(path,values):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps(v)+'\n' for v in values))
        row=dict(batch=1,indices=[1,2,3],query_sha256='same',update_calls={'backward':4,'optimizer':4},
            wall_seconds=10.,input_tokens=30,response_tokens=60,generation_seconds=6.,ppo_step_seconds=3.,
            reward_mean=2/3,eos_stopped=3,k3_kl=1.,loss=.2,peak_allocated_bytes=1024,
            allocator_ooms=0,allocator_retries=0)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            write(output/'EXPERIMENT.json',dict(source_checkpoints={'baseline':{'adapter_sha256':'a'},
                'odb':{'adapter_sha256':'b'}},batches=2,probe_samples=2))
            for name,checksum,flags in [('baseline','a',[True,False]),('odb','b',[True,True])]:
                root=output/name
                write(root/'REQUESTED_CONFIG.json',dict(sft_adapter_sha256=checksum))
                write(root/'runs/short/PARAMETER_AUDIT.json',dict(starting_adapter_sha256=checksum))
                records=[row,dict(row,batch=2)] if name=='baseline' else [row]
                lines(root/'runs/short/batches.jsonl',records)
                for stage in ['before','after']:
                    hits=flags if stage=='after' else [True,False]
                    write(root/f'probe_{stage}/RESULT.json',dict(indices=[10,11],samples=2,
                        adapter_sha256=checksum,numeric_accuracy=sum(hits)/2,natural_eos_rate=1.,
                        mean_response_tokens=20))
                    lines(root/f'probe_{stage}/rows.jsonl',[
                        dict(training_index=i,correct=hit) for i,hit in zip([10,11],hits)])
            summarize(output)
            result=json.loads((output/'RESULT.json').read_text())
            self.assertEqual(result['common_prefix_batches'],1)
            self.assertEqual(result['cases']['baseline']['common_prefix']['input_plus_response_tokens_per_second'],9.)
            self.assertEqual(result['cases']['odb']['probe_accuracy_change_pp'],50.)
            self.assertEqual(result['paired_probes']['after']['odb_only_correct'],1)
            self.assertFalse(result['cases']['odb']['completed_requested_batches'])
            bad=dict(row,update_calls={'backward':3,'optimizer':3})
            with self.assertRaises(AssertionError):phase_stats([bad])


if __name__=='__main__':
    unittest.main()
