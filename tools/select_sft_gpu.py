"""Select an available BF16 GPU without allocating CUDA tensors."""

import csv
import os
import subprocess
import sys


def main():
    report = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,compute_cap,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    requested = os.environ.get('GPU')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    allowed = None if visible is None else set(visible.split(','))
    choices = []
    for fields in csv.reader(report.splitlines()):
        index, uuid, name, capability, free, utilization = (f.strip() for f in fields)
        if allowed is not None and index not in allowed and uuid not in allowed:
            continue
        if requested is not None and requested not in (index, uuid):
            continue
        if float(capability) >= 8 and int(free) >= 16384 and int(utilization) <= 10:
            choices.append((int(free), uuid, name, index))
    if not choices:
        print('No available BF16 GPU with >=16 GiB free and <=10% utilization. Current status:', file=sys.stderr)
        print(report, file=sys.stderr)
        raise SystemExit(1)
    free, uuid, name, index = max(choices)
    print(f'Selected GPU {index}: {name}, {free} MiB free ({uuid})', file=sys.stderr)
    print(uuid)


if __name__ == '__main__':
    main()
