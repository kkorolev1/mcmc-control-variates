import jax.numpy as jnp
from itertools import repeat
import pandas as pd


def fill_diagonal(a, val):
    assert a.ndim >= 2
    i, j = jnp.diag_indices(min(a.shape[-2:]))
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
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = (
            self._data.loc[key, "total"] / self._data.loc[key, "counts"]
        )

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()


def moving_average(x, w):
    return jnp.convolve(x, jnp.ones(w), "valid") / w
