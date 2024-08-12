from collections import defaultdict
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns


class Logger:
    def __init__(self):
        self.step = 0
        self.data = defaultdict(dict)

    def set_step(self, step):
        self.step = step

    def _scalar_name(self, scalar_name):
        return f"{scalar_name}"

    def add_scalar(self, scalar_name, scalar):
        self.data[self._scalar_name(scalar_name)][self.step] = scalar

    def to_pandas(self):
        series = [
            pd.Series(
                name=scalar_name,
                data=value_by_step.values(),
                index=value_by_step.keys(),
            )
            for scalar_name, value_by_step in self.data.items()
        ]
        return pd.concat(series, axis=1)


def plot_log_results(logger: Logger):
    log_results = logger.to_pandas()
    nplots = len(log_results.columns)
    ncols = math.floor(math.sqrt(nplots))
    nrows = math.ceil(nplots / ncols)
    # print(nrows, ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
    if axes.ndim == 1:
        axes = axes[:, None]

    for i in range(nplots):
        ax = axes[i // ncols, i % ncols]
        data = log_results.iloc[:, i]
        plot = sns.lineplot(data=data, ax=ax)
        if data.min() > 0:
            ax.set(yscale="log")

    fig.tight_layout()
    fig.show()
