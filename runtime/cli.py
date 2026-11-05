"""Prepare immutable inputs, launch stages, and resume without overwriting results."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import yaml
from project import PROJECT, CODE, VENDOR, read, write, sha, source_manifest


def profile(path):
    candidate=Path(path)
    if not candidate.is_file():candidate=PROJECT/'configs'/f'{path}.yaml'
    selected=yaml.safe_load(candidate.read_text(encoding='utf-8-sig'))
    common=yaml.safe_load((PROJECT/'configs/common.yaml').read_text(encoding='utf-8'))
    common['trl_algorithm_contract']=read(PROJECT/'configs/trl_contract.json')
    assert selected['name'] in ('baseline','optimized')
    assert selected['runtime']['rollout'] in ('native','merged_compiled')
    assert selected['sft']['checkpoint_policy'] in ('all','none')
    assert type(selected['runtime']['python_gc_per_batch']) is bool
    selected['common']=common
    return selected


def read_rows(path,count):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    from numeric_reward import numeric_answer
    assert len(rows)==count,(path,len(rows),count)
    assert all(isinstance(r['question'],str) and r['question'].strip() and
               isinstance(r['answer'],str) and numeric_answer(r['answer'],gold=True) is not None for r in rows)
    assert len({r['question'] for r in rows})==len(rows),'Duplicate questions'
    return rows


def prepare(args):
    from transformers import AutoTokenizer
    from llamafactory.hparams import DataArguments
    from llamafactory.data import get_template_and_fix_tokenizer
    from llamafactory.data.processor.unsupervised import UnsupervisedDatasetProcessor
    selected=profile(args.profile)
    if args.merge_backup:selected['runtime']['merge_backup']=args.merge_backup
    model=Path(args.model).expanduser().resolve()
    model_config=read(model/'config.json')
    assert model_config['model_type']=='qwen3' and model_config['num_hidden_layers']==36
    assert model_config['hidden_size']==2560 and model_config['tie_word_embeddings']
    assert not model_config.get('quantization_config'),'Unquantized BF16/FP32 base required'
    train=read_rows(args.train,7473);test=read_rows(args.test,1319)
    assert not ({r['question'] for r in train}&{r['question'] for r in test}),'Train/test overlap'
    root=Path(args.run_dir).expanduser().resolve()
    assert not root.exists(),'Choose a new run directory; prepare never overwrites'
    # Validate the external tokenizer before creating the run directory.
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    data_args=DataArguments(template='qwen3_nothink',cutoff_len=1024)
    template=get_template_and_fix_tokenizer(tokenizer,data_args)
    processor=UnsupervisedDatasetProcessor(template=template,tokenizer=tokenizer,processor=None,data_args=data_args)
    metadata=[]
    for index,row in enumerate(train):
        ids,_=processor._encode_data_example([dict(role='user',content=row['question'])],[],None,None,[],[],[])
        metadata.append(dict(index=index,prompt_tokens=len(ids),prompt_token_sha256=hashlib.sha256(
            json.dumps(ids,separators=(',',':')).encode()).hexdigest()))
    maximum=max(r['prompt_tokens'] for r in metadata)
    if selected['runtime']['rollout']!='native':
        assert maximum+512<=selected['runtime']['static_cache_length'],(maximum,'StaticCache capacity exceeded')
    root.mkdir(parents=True)
    data=root/'data';data.mkdir()
    for name,rows in [('train',train),('test',test)]:
        (data/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    for name,output in [('gsm8k_sft',True),('gsm8k_ppo',False)]:
        write(data/f'{name}.json',[dict(instruction=r['question'],input='',output=r['answer'] if output else '') for r in train])
    write(data/'dataset_info.json',{name:dict(file_name=f'{name}.json',columns=dict(prompt='instruction',query='input',response='output'))
                                    for name in ('gsm8k_sft','gsm8k_ppo')})
    manifest={name:dict(sha256=sha(data/f'{name}.jsonl'),examples=len(rows)) for name,rows in [('train',train),('test',test)]}
    write(data/'UPSTREAM_MANIFEST.json',manifest)
    (root/'PPO_PROMPT_METADATA.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in metadata),encoding='utf-8')
    weight_files=sorted(set(model.glob('*.safetensors'))|set(model.glob('pytorch_model*.bin')))
    assert weight_files,'No external model weights found'
    model_files=sorted(set(weight_files)|set(model.glob('*.json'))|set(model.glob('*.model'))|set(model.glob('*.tiktoken')))
    print('Hashing external model files (weights remain outside the project)...',flush=True)
    model_hashes={p.name:sha(p) for p in model_files}
    derived={p.relative_to(root).as_posix():sha(p) for p in
             [data/'gsm8k_sft.json',data/'gsm8k_ppo.json',data/'dataset_info.json',root/'PPO_PROMPT_METADATA.jsonl']}
    write(root/'INPUT_IDENTITY.json',dict(model=model_hashes,data=manifest,prepared_data=derived))
    reward=dict(model='GSM8K exact numeric answer rule (no learned reward model)',reward_type='exact_numeric_answer',
        correct_score=1.,incorrect_score=0.,train_sha256=manifest['train']['sha256'],
        scorer_source_sha256=sha(CODE/'numeric_reward.py'),training_labels_only=True,
        learned_reward_model_loaded=False,format_bonus=False,
        parsing='#### first, otherwise last boxed/final number; Decimal normalized equality')
    reward['revision']=hashlib.sha256(json.dumps(reward,sort_keys=True).encode()).hexdigest()
    write(root/'REWARD_SPEC.json',reward)
    config=dict(selected['common']['ppo'],model_name_or_path=str(model),dataset_dir=str(data),
        cache_dir=str(root/'cache'),output_dir=str(root/'runs/UNSTARTED/model'),
        ppo_batch_python_gc=selected['runtime']['python_gc_per_batch'],
        ppo_compiled_rollout=selected['runtime']['rollout']!='native')
    repair=dict(output_head_lora=True,trainable_lora_parameters=17751040,trainable_ppo_parameters=17753601,
        kl_estimator='k3: exp(log_ref-log_actor)-1-(log_ref-log_actor)',quality_zero_batch_stop=20,k3_stop=50,
        sampling='top_k0 top_p1 temperature1',generation_autocast='bf16',ppo_epochs=4)
    write(root/'REQUESTED_CONFIG.json',dict(model='Qwen3-4B-Base',requested_lf_arguments=config,repair=repair,
        sft_checkpoint=None,sft_adapter_sha256=None,raw_training_examples=7473,rollout_batches=2491,optimizer_updates=9964,
        fresh_ppo_optimizer=True,new_value_head=True,kl_reference='Frozen completed SFT adapter',reward=reward))
    sft=dict(selected['common']['sft'],model_name_or_path=str(model),dataset_dir=str(data),
        output_dir=str(root/'sft/model'),cache_dir=str(root/'cache_sft'))
    # Enable the native Trainer input-grad hook in both modes. Individual layer
    # checkpoint flags are selected in training_step, after Trainer initialization.
    write(root/'SFT_CONFIG.json',sft)
    write(root/'DATA_PREFLIGHT.json',dict(passed=True,train_examples=7473,test_examples=1319,
                                        max_ppo_prompt_tokens=maximum,formal_max_new_tokens=512))
    write(root/'PROFILE.json',selected)
    write(root/'RUNTIME_BUILD.json',source_manifest())
    print(json.dumps(dict(prepared=str(root),profile=selected['name'],max_ppo_prompt_tokens=maximum),indent=2))


def verify_run(root):
    assert read(root/'RUNTIME_BUILD.json')==source_manifest(),'Source/config changed after prepare; use a fresh directory'
    identity=read(root/'INPUT_IDENTITY.json')
    request=read(root/'REQUESTED_CONFIG.json')
    model=Path(request['requested_lf_arguments']['model_name_or_path'])
    for name,checksum in identity['model'].items():
        assert sha(model/name)==checksum,('External model changed',name)
    for name,item in identity['data'].items():assert sha(root/f'data/{name}.jsonl')==item['sha256']
    for name,checksum in identity['prepared_data'].items():assert sha(root/name)==checksum,('Prepared data changed',name)


def attach_sft(root,path):
    path=Path(path).expanduser().resolve()
    assert (path/'adapter_config.json').is_file() and (path/'adapter_model.safetensors').is_file()
    config=read(path/'adapter_config.json')
    assert config['r']==8 and config['lora_alpha']==16 and config['lora_dropout']==0
    assert set(config['target_modules'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head'}
    request=read(root/'REQUESTED_CONFIG.json');checksum=sha(path/'adapter_model.safetensors')
    if (root/'runs').exists() and any((root/'runs').iterdir()):
        assert request['sft_adapter_sha256']==checksum,'Cannot replace reference after PPO has started'
    request.update(sft_checkpoint=str(path),sft_adapter_sha256=checksum)
    request['requested_lf_arguments']['adapter_name_or_path']=str(path)
    write(root/'REQUESTED_CONFIG.json',request)


def launch(root,script,*arguments):
    env=os.environ.copy()
    env.update(PPO4_RUN_ROOT=str(root),PYTHONPATH=str(CODE)+os.pathsep+str(VENDOR),
        DISABLE_VERSION_CHECK='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
    subprocess.run([sys.executable,str(CODE/script),*map(str,arguments)],env=env,check=True)


def run_ppo(root,args):
    from urllib.request import urlopen
    selected=read(root/'PROFILE.json');name=args.name
    assert name and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name)
    assert not (root/'runs'/name).exists(),'PPO output exists; choose a new --name'
    if args.sft:attach_sft(root,args.sft)
    request=read(root/'REQUESTED_CONFIG.json')
    assert request['sft_checkpoint'],'Run SFT first, or pass --sft /path/to/completed/adapter'
    assert sha(Path(request['sft_checkpoint'])/'adapter_model.safetensors')==request['sft_adapter_sha256']
    ready=root/f'REWARD_SERVER_{name}.json'
    assert not ready.exists(),'Reward-server record exists; choose a new --name'
    env=os.environ.copy();env.update(PPO4_RUN_ROOT=str(root),PYTHONPATH=str(CODE)+os.pathsep+str(VENDOR))
    log=(root/f'reward_{name}.stdout.log').open('x')
    child=subprocess.Popen([sys.executable,str(CODE/'reward_server.py'),'--name',name],env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        deadline=time.monotonic()+30
        while not ready.exists():
            if child.poll() is not None:raise RuntimeError('Reward server exited; inspect its stdout log')
            if time.monotonic()>deadline:raise TimeoutError('Reward server readiness timeout')
            time.sleep(.1)
        url=read(ready)['url']
        with urlopen(url.replace('/reward','/'),timeout=5) as response:assert json.load(response)['ready']
        extra=['--stress'] if args.stress else []
        if args.resume_state:extra+=['--resume-state',str(Path(args.resume_state).resolve())]
        if args.strict:extra+=['--strict']
        launch(root,'train_ppo.py','--name',name,'--variant',
            'native' if selected['runtime']['rollout']=='native' else 'compiled',
            '--reward-url',url,'--limit',args.limit,'--state-save-every',args.save_every,*extra)
    finally:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        log.close()


def main():
    os.environ.setdefault('DISABLE_VERSION_CHECK','1')
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    c=sub.add_parser('config');c.add_argument('--profile',default='baseline')
    d=sub.add_parser('doctor');d.add_argument('--gpu',action='store_true')
    p=sub.add_parser('prepare')
    p.add_argument('--profile',default='baseline');p.add_argument('--run-dir',required=True)
    for key in ('model','train','test'):p.add_argument('--'+key,required=True)
    p.add_argument('--merge-backup',choices=['gpu','cpu'])
    for command in ('sft','ppo','quality','evaluate','all','verify'):
        item=sub.add_parser(command);item.add_argument('--run-dir',required=True)
        if command in ('sft','all'):
            item.add_argument('--sft-limit',type=int,default=935);item.add_argument('--sft-name',default='sft');item.add_argument('--resume-hf')
        if command in ('ppo','all'):
            item.add_argument('--name',default='full');item.add_argument('--limit',type=int,default=2491)
            item.add_argument('--save-every',type=int,default=250);item.add_argument('--sft')
            item.add_argument('--resume-state');item.add_argument('--strict',action='store_true');item.add_argument('--stress',action='store_true')
        if command in ('quality','evaluate'):
            item.add_argument('--adapter',required=True)
            item.add_argument('--name' if command=='quality' else '--label',required=True)
    args=parser.parse_args()
    if args.command=='config':print(json.dumps(profile(args.profile),indent=2));return
    if args.command=='doctor':
        from doctor import main as doctor
        doctor(args.gpu);return
    if args.command=='prepare':prepare(args);return
    root=Path(args.run_dir).expanduser().resolve();verify_run(root)
    if args.command=='verify':
        print(json.dumps(dict(passed=True,source_and_external_input_hashes_verified=True,run_dir=str(root))));return
    if args.command in ('sft','all'):
        assert 0<args.sft_limit<=935
        extra=['--resume-hf',args.resume_hf] if args.resume_hf else []
        launch(root,'train_sft.py','--limit',args.sft_limit,'--name',args.sft_name,*extra)
        if args.sft_limit==935:attach_sft(root,root/args.sft_name/'model')
        if args.command=='sft':return
        assert args.sft_limit==935 and args.limit==2491 and not args.stress,'all requires complete training'
        launch(root,'quality_check.py','--adapter',read(root/'REQUESTED_CONFIG.json')['sft_checkpoint'],'--name','sft_ready')
    if args.command in ('ppo','all'):
        assert 0<args.limit<=2491 and args.save_every>0
        run_ppo(root,args)
        if args.command=='ppo':return
        final=root/'runs'/args.name/'model'
        launch(root,'quality_check.py','--adapter',final,'--name','ppo_final')
        launch(root,'evaluate_gsm8k.py','--adapter',final,'--label','PPO')
        return
    if args.command=='quality':launch(root,'quality_check.py','--adapter',args.adapter,'--name',args.name)
    if args.command=='evaluate':launch(root,'evaluate_gsm8k.py','--adapter',args.adapter,'--label',args.label)
