"""Choose the BF16 GPU with most free memory; no CUDA allocation or reservation."""
import csv
import json
import os
import subprocess
import sys


def select(report, visible=None, requested=None, minimum_mib=24576):
    allowed = None if visible is None else set(visible.split(','))
    choices = []
    for fields in csv.reader(report.splitlines()):
        index, uuid, name, capability, free, utilization = (x.strip() for x in fields)
        if allowed is not None and index not in allowed and uuid not in allowed:
            continue
        if requested is not None and requested not in (index, uuid):
            continue
        if float(capability) >= 8 and int(free) >= minimum_mib:
            choices.append((int(free), -int(utilization), uuid, name, index))
    if not choices:
        raise ValueError(f'No visible BF16 GPU with >= {minimum_mib} MiB free.\n{report}')
    return max(choices)


if __name__ == '__main__':
    report = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,name,compute_cap,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    try:
        free, neg_util, uuid, name, index = select(report, os.environ.get('CUDA_VISIBLE_DEVICES'),
            os.environ.get('GPU'), int(os.environ.get('MIN_FREE_MIB', '24576')))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(dict(selected_gpu=index, uuid=uuid, name=name, free_mib=free,
                         utilization=-neg_util)), file=sys.stderr)
    print(uuid)
