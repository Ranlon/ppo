"""Optional external data export. No dataset is shipped with this project."""
from pathlib import Path
import argparse
import json
from datasets import load_dataset

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
parser.add_argument('--revision',default='3101c7d5072418e28b9008a6636bde82a006892c')
args=parser.parse_args()
root=Path(args.output).expanduser().resolve();root.mkdir(parents=True,exist_ok=False)
data=load_dataset('openai/gsm8k','main',revision=args.revision)
for split in ('train','test'):
    (root/f'{split}.jsonl').write_text(''.join(json.dumps(dict(question=row['question'],answer=row['answer']),ensure_ascii=False)+'\n' for row in data[split]),encoding='utf-8')
print(str(root))
