"""Loopback-only rule scorer; imports no Torch or learned reward model."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import argparse
import hashlib
import json
import os
import threading
import time
from numeric_reward import ExactNumericReward
from project import run_root
ROOT=run_root()


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*a):pass
    def do_GET(self):
        body=json.dumps(dict(ready=True,**self.server.reward.manifest)).encode()
        self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        try:
            assert self.path=='/reward'
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with self.server.lock:
                result=self.server.reward.score(payload['prompts'],payload['responses'],payload['indices'])
                record=dict(unix=time.time(),run=payload.get('run'),indices=payload['indices'],**result,
                    response_sha256=[hashlib.sha256(s.encode()).hexdigest() for s in payload['responses']])
                with self.server.audit.open('a') as handle:handle.write(json.dumps(record)+'\n')
            body=json.dumps(result).encode();self.send_response(200)
        except Exception as exc:
            body=json.dumps(dict(error=str(exc))).encode();self.send_response(400)
        self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='main');args=parser.parse_args()
    reward=ExactNumericReward(ROOT)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    server.reward=reward;server.lock=threading.Lock();server.audit=ROOT/('reward_'+args.name+'.jsonl')
    ready=dict(pid=os.getpid(),url=f'http://127.0.0.1:{server.server_port}/reward',model=reward.manifest['model'],
        revision=reward.manifest['revision'],reward_type='exact_numeric_answer',learned_reward_model_loaded=False)
    (ROOT/('REWARD_SERVER_'+args.name+'.json')).write_text(json.dumps(ready,indent=2)+'\n')
    print(json.dumps(ready),flush=True);server.serve_forever()


if __name__=='__main__':main()
