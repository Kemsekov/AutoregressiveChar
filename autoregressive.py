import math
from typing import Literal

#both implement re-zero approach
from kemsekov_torch.attention import SelfAttention
from kemsekov_torch.gated_delta_2 import GatedDelta2Scan
from kemsekov_torch.recurrent_layer import RecurrentLayer

from kemsekov_torch.common_modules import (
    Residual, Transpose, SwiGLU,ConcatTensors,SumTensors,
    StepSequential,StepState,init_module_state,step_module
)
import torch
import torch.nn as nn
import torch.nn.functional as F

# module to convert text tokens to vector
class Embedding(nn.Module):
    """
    Module for token to embedding vector learning
    """
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(vocab_size, embedding_size))
        self.bias = nn.Parameter(torch.Tensor(embedding_size))

        self.reset_parameters()

    #normal init
    def reset_parameters(self):
        # Initialize weights with a normal distribution
        std = 1.0 / (self.vocab_size**0.5)
        
        nn.init.normal_(self.weight, mean=0.0, std=std)
        # Initialize bias to zeros
        nn.init.zeros_(self.bias)
        
    def forward(self, input):
        # Input is expected to be a tensor of indices
        return torch.nn.functional.embedding(input, self.weight)

    def encode(self,ind): return self(ind)
    
    def decode(self,act):
        return act@self.weight.T
    
class AutoregressiveChar(nn.Module):
    def __init__(
        self,
        vocab_size,
        internal_dim,
        layers=3,
        mlp_factor=4,
        heads=8,
        impl:Literal['attn','gd2']="attn",
        merge_implementation:Literal['concat','sum']='sum',
        recurrence:int|None=None,
    ):
        super().__init__()

        def mlp():
            return Residual([
                nn.RMSNorm(internal_dim),
                SwiGLU(internal_dim,internal_dim*mlp_factor),
                nn.Linear(internal_dim*mlp_factor,internal_dim),
            ])
            
        def get_imp():
            if impl=='attn':
                return StepSequential(
                    Transpose(1,-1),
                    SelfAttention(
                        internal_dim,
                        heads=heads,
                        kv_heads=heads//2,
                        head_dim=64,
                        add_alibi=True,
                        prenorm='rms',
                        is_causal=True,
                        dimensions=1,
                        xsa=True
                    ),
                    Transpose(1,-1),
                    mlp()
                )
            if impl=='gd2':
                return StepSequential(
                    GatedDelta2Scan(
                        dim=internal_dim,
                        heads=heads,
                        kv_heads=heads//2,
                        QK_dim=64,
                        V_dim=64
                    ),
                    mlp()
                )
        
        self.emb = Embedding(
            vocab_size,embedding_size=internal_dim
        )
        
        if merge_implementation=='concat':
            self.merge_activations=nn.Sequential(
                ConcatTensors(-1),
                nn.Linear(internal_dim*2,internal_dim,bias=False)
            )
        if merge_implementation=='sum':
            self.merge_activations=SumTensors()
        
        def get_layer():
            imp = get_imp()
            if recurrence is None:
                return imp
            return RecurrentLayer(imp,internal_dim,max_recurrence=recurrence)

        self.middle=StepSequential(*[
            get_layer()
            for i in range(layers)
        ])
        
    def decode(self,x):
        return self.emb.decode(x)
    
    def encode(self,x):
        return self.emb.encode(x)
    
    def forward(self,ind,previous_activations = None):
        x = self.encode(ind)
        previous_activations=x*0 if previous_activations is None else previous_activations
        
        x=self.merge_activations([x,previous_activations])
        x = self.middle(x)
        return x,self.decode(x)

    def init_state(self, batch_size, device=None, dtype=None):
        """
        Creates the recurrent state required by :meth:`step` / :meth:`generate`.

        The state is opaque: it is a nested structure mirroring the model
        layers, each holding whatever its implementation needs (KV cache for
        attention, memory matrix for gated delta, per-application caches for
        recurrent wrappers, ...). The tree is discovered by reflecting over
        ``self.middle``, so new wrappers (e.g. ``RecurrentLayer``,
        ``AttentionResidual``) compose automatically as long as they implement
        ``init_state``/``step``; a wrapper with stateful children that forgets
        to implement them raises instead of silently running full-sequence
        ``forward``.
        """
        return init_module_state(self.middle,batch_size,device=device,dtype=dtype)

    def step(self, ind, state=None):
        """
        Processes the next chunk of tokens `ind` (`[B]` or `[B, L]` ids) with
        the incremental state and returns `(activations, logits, state)`.

        When `state` is None a fresh state is created, so
        `step(ids)` can also be used as a one-shot forward replacement for a
        single token.
        """
        if ind.dim()==1:
            ind = ind[:,None]
        if state is None:
            state = self.init_state(ind.shape[0], device=ind.device, dtype=self.emb.weight.dtype)
        x = self.encode(ind)
        x = self.merge_activations([x, torch.zeros_like(x)])
        x, state = step_module(self.middle,x,state)
        return x, self.decode(x), state

    def generate(self, ind, max_new_tokens, temp=0.7, top_p=0.9):
        """
        Autoregressive generation with incremental state.

        ind: `[B, L]` (or `[L]`) prompt token ids.
        max_new_tokens: how many tokens to sample.

        Yields the freshly sampled token ids (`[B]` tensor) one by one, so the
        caller can stream them:

            for token in model.generate(prompt, 128):
                ...
        """
        if not isinstance(ind,torch.Tensor):
            ind = torch.tensor(ind)
        if ind.dim()==1:
            ind = ind[None]
        ind = ind.to(next(self.parameters()).device)

        with torch.no_grad():
            state = self.init_state(ind.shape[0], device=ind.device, dtype=self.emb.weight.dtype)
            # prefill the prompt in one parallel chunk
            x = self.encode(ind)
            x = self.merge_activations([x, torch.zeros_like(x)])
            x, state = step_module(self.middle,x,state)
            logits = self.decode(x)[:,-1]
            for _ in range(max_new_tokens):
                next_token = sample(logits,temp=temp,top_p=top_p)
                yield next_token
                # one incremental step per new token
                x = self.encode(next_token[:,None])
                x = self.merge_activations([x, torch.zeros_like(x)])
                x, state = step_module(self.middle,x,state)
                logits = self.decode(x)[:,-1]

    def params_count(self):
        return sum([p.numel() for p in self.parameters()])


def sample(logits: torch.Tensor, temp: float = 0.7, top_p: float = 0.9) -> torch.Tensor:
    """
    Samples class indices from a batch of logits using temperature scaling and Top-P (Nucleus) filtering.
    
    Args:
        logits (torch.Tensor): Tensor of shape [BATCH, classes]
        temp (float): Temperature for scaling. 0.0 performs greedy argmax.
        top_p (float): Nucleus sampling threshold (0.0 < top_p <= 1.0). 
                       1.0 turns off Top-P filtering.
    """
    # 1. Handle greedy selection if temperature is 0
    if temp == 0.0:
        return torch.argmax(logits, dim=-1)
        
    # 2. Scale the logits by the temperature
    scaled_logits = logits / temp
    
    # 3. Apply Top-P (Nucleus) filtering if top_p < 1.0
    if top_p < 1.0:
        # Sort probabilities/logits in descending order
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        # We shift the mask by 1 to make sure we keep the first token that exceeds top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Mask out excluded logits by setting them to negative infinity
        # This gives them 0 probability during the final softmax step
        sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))
        
        # Scatter the filtered logits back to their original position mapping
        scaled_logits = torch.empty_like(sorted_logits).scatter_(-1, sorted_indices, sorted_logits)

    # 4. Final softmax and multinomial sampling
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
