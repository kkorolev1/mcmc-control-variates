import jax.numpy as np
from itertools import repeat

def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = np.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader
