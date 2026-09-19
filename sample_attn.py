import torch
from autoregressive import AutoregressiveChar,sample
from kemsekov_torch.train import load_last_checkpoint
torch.manual_seed(0)

tokenizer=torch.load("tokenizer.pt",weights_only=False)

model = AutoregressiveChar(tokenizer.vocab_size,256,layers=1,mlp_factor=1,impl='attn')
model=load_last_checkpoint(model,"runs/test-autoregressive-attn-padmask").eval().cuda()


start = "Some days ago"
ids = tokenizer.encode(start).tolist()

to_generate=128

for i in range(to_generate):
    with torch.no_grad():
        act,logits = model(torch.tensor([ids],device='cuda'))
        next_token = sample(logits[:,-1],temp=0.7).item()
        ids.append(next_token)
ids=torch.tensor(ids)
print(tokenizer.decode(ids))
