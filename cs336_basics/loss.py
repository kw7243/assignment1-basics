import math
import torch
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.softmax import softmax

def cross_entropy(logits: Float[Tensor, " ... vocab_size"], 
                  targets: Int[Tensor, "... "]) -> Float[Tensor, ""]: 
    """
    -E[logq(y|x)] ~= -sum(logq(yi|xi)) = -sum(log func(yi,xi))

    (B,L,D) --> ()

    Args:
        logits (Float[Tensor, "... vocab_size"]): Unnormalized scores. (ex. (B,L,V))
        targets (Int[Tensor, "... "]): ground truth (t+1)th next token for t_th logit

    Returns:
        Float[Tensor, "..."]: average CE over B,L dim; shape = ()
                                avg cross-entropy term per position t = -logq(yi | xi) = -log (Π pk_theta(xi) ^ 1(k == yi))
                                averaged over t

    Requires:
        dim selects a nonempty dimension. Each slice has a finite maximum.
    """
    logits = logits - logits.amax(dim=-1, keepdim=True)
    exp = math.e**logits # (B,L,V)

    logits_target = torch.gather(input=logits, 
                               dim=-1, 
                               index=targets.unsqueeze(-1)).squeeze(-1) # (B,L,V) --> (B,L), q(yi|xi) = Π pk^(1(k=target))

    # For just (V,), after indexing,
    # log likelihood i = log softmax(o_i)[yi] = log(exp(o_{i,yi})/sum(exp(o_k)))
    # = o_{i,yi} - log sum(exp(o_k))
    log_likelihoods = logits_target - torch.log(exp.sum(dim=-1))

    return -log_likelihoods.mean()





