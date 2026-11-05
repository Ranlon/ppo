"""Environment report and selected legacy compatibility checks."""
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys

def main(gpu=False):
    import torch
    required={'torch':'2.6.0','transformers':'4.57.1','peft':'0.17.1','trl':'0.9.6','accelerate':'1.11.0',
              'datasets':'4.0.0','triton':'3.2.0','numpy':'1.26.4'}
    versions={name:metadata.version(name) for name in required}
    for name,want in required.items():assert versions[name].split('+')[0]==want,(name,versions[name],want)
    assert sys.version_info[:2]==(3,11)
    report=dict(python=sys.version,platform=platform.platform(),versions=versions,
                cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),
                native_windows_supported=False)
    if gpu:
        assert platform.system()=='Linux','Use Linux or WSL2; native Windows Triton was not validated'
        assert torch.cuda.device_count()==1,'Expose exactly one GPU with CUDA_VISIBLE_DEVICES'
        assert torch.cuda.is_bf16_supported()
        props=torch.cuda.get_device_properties(0)
        report.update(gpu=props.name,compute_capability=[props.major,props.minor],total_vram_gib=props.total_memory/2**30)
        assert props.major<10,'Blackwell needs a newer separately validated PyTorch/CUDA stack'
    check=subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True)
    report.update(pip_check_exit=check.returncode,pip_check=check.stdout.strip()+check.stderr.strip())
    print(json.dumps(report,indent=2))
    if check.returncode:raise RuntimeError('Dependency check failed')
