import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.

        LongTensors preferred since they can always be downcasted to match transformer dtype
    """
    n = len(dataset)
    starts = np.random.randint(0, n - context_length, batch_size) # (B,)
    offsets = np.arange(0, context_length) # (L,)

    indices = starts[:, None] + offsets[None, :] # (B,L)

    inputs = dataset[indices] 
    targets = dataset[indices + 1]

    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)
