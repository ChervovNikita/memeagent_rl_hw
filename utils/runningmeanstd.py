

import numpy as np


class RunningMeanStd:
    """
    Computes running mean and std.
    """

    def __init__(self):
        # Keep numerically sensitive running stats in float64 to avoid overflow in long runs.
        self.mean = np.zeros((1,), dtype=np.float64)
        self.var = np.ones((1,), dtype=np.float64)
        self.count = 0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return

        # Avoid propagating infinities/nans from upstream signals.
        x = np.nan_to_num(x, nan=0.0, posinf=1e12, neginf=-1e12)

        # Clip extreme outliers to keep second moment in a representable range.
        x = np.clip(x, -1e12, 1e12)

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)

        # update count
        n = int(x.shape[0]) if x.ndim > 0 else 1
        if n <= 0:
            return
        old_count = self.count
        self.count += n

        # update mean
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count

        # update var
        m_a = self.var * old_count
        m_b = batch_var * n
        M2 = m_a + m_b + np.square(delta) * n
        self.var = M2 / self.count
        self.var = np.maximum(self.var, 1e-12)

    def mean(self):
        assert not np.isnan(self.mean).any()
        return self.mean.item()

    def std(self):
        assert not np.isnan(self.var).any()
        return np.sqrt(self.var).item()

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


