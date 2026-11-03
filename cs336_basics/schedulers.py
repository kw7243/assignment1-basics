
import math

def get_lr_cosine_schedule(
    t: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Compute the learning rate using linear warmup and cosine decay.

    Args:
        t: Current training iteration, starting at 0.
        max_learning_rate: Maximum learning rate.
        min_learning_rate: Final learning rate.
        warmup_iters: Number of warmup iterations.
        cosine_cycle_iters: Iteration at which cosine decay ends.

    Returns:
        Learning rate at iteration t.

    Requires:
        t >= 0.
        0 <= min_learning_rate <= max_learning_rate.
        0 <= warmup_iters < cosine_cycle_iters.

    Effects:
        None. This function does not modify optimizer state.
    """

    # Phase 1: Linear warmup
    if t < warmup_iters:
        return t*max_learning_rate/warmup_iters

    # Phase 2: Cosine decay
    if t <= cosine_cycle_iters:
        return (min_learning_rate + 0.5*(1+ math.cos(math.pi*(t - warmup_iters)/(cosine_cycle_iters - warmup_iters)))
                *(max_learning_rate - min_learning_rate))

    # Phase 3: Constant minimum learning rate
    return min_learning_rate
