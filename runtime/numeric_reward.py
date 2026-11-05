"""GSM8K training-only exact numeric reward, with no learned reward model."""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
import time


def numeric_answer(text,gold=False):
    text=text.split('</think>')[-1]
    text=re.sub(r'<\|[^>]+\|>','',text).strip()
    if '####' in text:
        match=re.match(r'\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)',text.rsplit('####',1)[1])
    elif gold:
        return None
    else:
        boxed=re.findall(r'\\boxed\{\s*\$?([-+]?\d[\d,]*(?:\.\d+)?)\s*\}',text)
        values=boxed or re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?',text)
        match=re.fullmatch(r'([-+]?\d[\d,]*(?:\.\d+)?)',values[-1] if values else '')
    if not match:return None
    try:
        value=Decimal(match.group(1).replace(',',''))
        return str(value.normalize()) if value.is_finite() else None
    except InvalidOperation:return None


class ExactNumericReward:
    def __init__(self,root):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'REWARD_SPEC.json').read_text())
        path=self.root/'data/train.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==self.manifest['train_sha256']
        assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==self.manifest['scorer_source_sha256']
        self.rows=[json.loads(line) for line in path.read_text().splitlines()]
        self.targets=[numeric_answer(row['answer'],gold=True) for row in self.rows]
        assert len(self.rows)==7473 and all(value is not None for value in self.targets)

    def score(self,prompts,responses,indices):
        started=time.perf_counter()
        assert len(prompts)==len(responses)==len(indices)>0
        scores=[];predictions=[];targets=[]
        for prompt,response,index in zip(prompts,responses,indices):
            assert type(index) is int and 0<=index<len(self.rows),'Unknown training index'
            assert prompt==self.rows[index]['question'],'Question does not match the training index'
            assert isinstance(response,str)
            prediction=numeric_answer(response);target=self.targets[index]
            scores.append(float(prediction is not None and prediction==target))
            predictions.append(prediction);targets.append(target)
        return dict(scores=scores,predictions=predictions,targets=targets,
            seconds=time.perf_counter()-started,input_tokens=0,reward_type='exact_numeric_answer',
            model=self.manifest['model'],revision=self.manifest['revision'],learned_reward_model_loaded=False)
