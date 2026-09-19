from typing import Literal

#both implement re-zero approach
from kemsekov_torch.attention import SelfAttention
from kemsekov_torch.gated_delta_2 import GatedDelta2Scan

from kemsekov_torch.common_modules import Residual, Transpose, SwiGLU,ConcatTensors,SumTensors
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
        output = torch.nn.functional.embedding(input, self.weight) + self.bias
        return output

class AutoregressiveChar(nn.Module):
    def __init__(
        self,
        vocab_size,
        internal_dim,
        layers=3,
        mlp_factor=4,
        heads=8,
        impl:Literal['attn','gd2']="attn",
        merge_implementation:Literal['concat','sum']='sum'
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
                return nn.Sequential(
                    Transpose(1,-1),
                    SelfAttention(
                        internal_dim,
                        heads=heads,
                        kv_heads=heads//2,
                        head_dim=64,
                        add_absolute_pos=True,
                        prenorm='rms',
                        is_causal=True,
                        dimensions=1
                    ),
                    Transpose(1,-1),
                    mlp()
                )
            if impl=='gd2':
                return nn.Sequential(
                    GatedDelta2Scan(
                        dim=internal_dim,
                        heads=heads,
                        kv_heads=heads//2,
                        QK_dim=64,
                        V_dim=64
                    ),
                    mlp()
                )
        self.encode = Embedding(
            vocab_size,embedding_size=internal_dim
        )
        
        if merge_implementation=='concat':
            self.merge_activations=nn.Sequential(
                ConcatTensors(-1),
                nn.Linear(internal_dim*2,internal_dim,bias=False)
            )
        if merge_implementation=='sum':
            self.merge_activations=SumTensors()
        
        self.middle=nn.Sequential(*[
            get_imp()
            for i in range(layers)
        ])
        self.decode=nn.Linear(
            internal_dim,vocab_size,bias=False
        )
        
    def forward(self,ind,previous_activations = None):
        x = self.encode(ind)
        previous_activations=x*0 if previous_activations is None else previous_activations
        
        x=self.merge_activations([x,previous_activations])
        x = self.middle(x)
        return x,self.decode(x)
    
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
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        
        # Scatter the filtered logits back to their original position mapping
        scaled_logits = torch.gather(sorted_logits, dim=-1, index=sorted_indices.argsort(dim=-1))

    # 4. Final softmax and multinomial sampling
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
