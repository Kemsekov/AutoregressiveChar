import argparse
import time
import torch
from autoregressive import AutoregressiveChar
from kemsekov_torch.train import load_last_checkpoint

# Model internal dim
internal_dim=256

# how many times repeat in a loop same layer. integer or None
reccurence=None

#model layers
layers=3

#model MLP factor
mlp_factor=4

checkpoint_path = 'runs/test-autoregressive-gd2'



parser=argparse.ArgumentParser()
parser.add_argument("--prompt",default="Some days ago")
parser.add_argument("--to_generate",type=int,default=1024)
parser.add_argument("--seed",type=int,default=None)
args=parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

tokenizer=torch.load("tokenizer.pt",weights_only=False)

# model = AutoregressiveChar(tokenizer.vocab_size,256,layers=1,mlp_factor=1,impl='gd2')
model = AutoregressiveChar(tokenizer.vocab_size,internal_dim,layers=layers,mlp_factor=mlp_factor,impl='gd2',recurrence=reccurence)

model=load_last_checkpoint(model,checkpoint_path).eval().cuda()

ids = tokenizer.encode(args.prompt).tolist()

start_time=time.time()
with torch.inference_mode():
    prompt=torch.tensor([ids],device='cuda')
    print(tokenizer.decode(prompt[0].cpu()),end="",flush=True)
    for t in model.generate(prompt,args.to_generate,temp=0.7):
        print(tokenizer.decode(t.cpu()),end="",flush=True)
print() #add newline at the end
elapsed=time.time()-start_time
print(f"generated {args.to_generate} tokens in {elapsed:.3f}s ({args.to_generate/elapsed:.1f} tok/s)")
