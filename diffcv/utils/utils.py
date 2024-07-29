import jax.numpy as np
from itertools import repeat
import pandas as pd


def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = np.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0
    
    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]
    
    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()