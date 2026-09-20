import argparse
import time
import torch
from autoregressive import AutoregressiveChar
from kemsekov_torch.train import load_last_checkpoint

parser=argparse.ArgumentParser()
parser.add_argument("--prompt",default="Some days ago")
parser.add_argument("--to_generate",type=int,default=128)
parser.add_argument("--seed",type=int,default=None)
args=parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

tokenizer=torch.load("tokenizer.pt",weights_only=False)

# model = AutoregressiveChar(tokenizer.vocab_size,256,layers=1,mlp_factor=1,impl='attn')
model=AutoregressiveChar(tokenizer.vocab_size,256,layers=3,mlp_factor=4,impl='attn')
model=load_last_checkpoint(model,"runs/test-autoregressive-attn-alibi").eval().cuda()


ids = tokenizer.encode(args.prompt).tolist()

start_time=time.time()
with torch.no_grad():
    prompt=torch.tensor([ids],device='cuda')
    print(tokenizer.decode(prompt[0].cpu()),end="",flush=True)
    for t in model.generate(prompt,args.to_generate,temp=0.7):
        print(tokenizer.decode(t.cpu()),end="",flush=True)
print() #add newline at the end
elapsed=time.time()-start_time
print(f"generated {args.to_generate} tokens in {elapsed:.3f}s ({args.to_generate/elapsed:.1f} tok/s)")
