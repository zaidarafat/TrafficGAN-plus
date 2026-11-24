"""Wrappers for third-party baselines.

CTGAN and CopulaGAN: SDV (pip install sdv), trained on flow records in the original feature space with
the class label as a categorical column; conditional sampling per class. Generated flows are grouped into
windows of T in generation order.
DoppelGANger: gretel-synthetics PyTorch DGAN (pip install gretel-synthetics), one model per class on windows."""
import numpy as np
import pandas as pd
import torch


def _flow_frame(prep, X_windows, y_windows, max_rows, seed):
    rng = np.random.default_rng(seed)
    d = X_windows.shape[-1]; T = X_windows.shape[1]
    O = prep.inverse(X_windows).reshape(-1, d)
    y = np.repeat(y_windows, T)
    if max_rows and len(O) > max_rows:                       # class-stratified subsample (keeps proportions)
        idx = rng.choice(len(O), max_rows, replace=False)
        # guarantee every class is represented
        for k in np.unique(y):
            if not np.any(y[idx] == k):
                idx = np.concatenate([idx, rng.choice(np.where(y == k)[0], min(1000, (y == k).sum()), False)])
        O, y = O[idx], y[idx]
    cols = [f"f{j}" for j in range(d)]
    df = pd.DataFrame(O, columns=cols)
    df["label"] = [prep.classes[k] for k in y]
    for j in np.where(prep.discrete)[0]:
        df[cols[j]] = df[cols[j]].round().astype("int64")
    return df, cols


class SDVBaseline:
    def __init__(self, kind, prep, epochs=300, seed=0, max_rows=200_000):
        self.kind, self.prep, self.epochs, self.seed, self.max_rows = kind, prep, epochs, seed, max_rows

    def fit(self, X_windows, y_windows, log=None):
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer
        df, self.cols = _flow_frame(self.prep, X_windows, y_windows, self.max_rows, self.seed)
        md = SingleTableMetadata(); md.detect_from_dataframe(df)
        md.update_column("label", sdtype="categorical")
        for j, c in enumerate(self.cols):
            md.update_column(c, sdtype="categorical" if self.prep.discrete[j] else "numerical")
        cls = CTGANSynthesizer if self.kind == "ctgan" else CopulaGANSynthesizer
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        import inspect
        gpu_kw = "enable_gpu" if "enable_gpu" in inspect.signature(cls.__init__).parameters else "cuda"
        self.m = cls(md, epochs=self.epochs, enforce_min_max_values=True, verbose=bool(log),
                     **{gpu_kw: torch.cuda.is_available()})
        self.m.fit(df)
        return self

    def sample_windows(self, labels):
        from sdv.sampling import Condition
        T, d = self.prep.T, len(self.cols)
        out = np.zeros((len(labels), T, d), np.float32)
        for k in np.unique(labels):
            pos = np.where(labels == k)[0]
            n = len(pos) * T
            s = self.m.sample_from_conditions([Condition(num_rows=n, column_values={"label": self.prep.classes[k]})],
                                              max_tries_per_batch=200, batch_size=min(n, 100_000))
            X = s[self.cols].to_numpy(np.float64)
            if len(X) < n:                                       # rejection sampling may return fewer rows
                X = X[np.random.default_rng(self.seed).integers(0, len(X), n)]
            out[pos] = self.prep.scale(X[:n]).reshape(len(pos), T, d)
        return out


class DGANBaseline:
    """One DoppelGANger model per class (features only, no attributes)."""

    def __init__(self, T, d, epochs=200, seed=0):
        self.T, self.d, self.epochs, self.seed, self.models = T, d, epochs, seed, {}

    def fit(self, X_windows, y_windows, log=None):
        from gretel_synthetics.timeseries_dgan.dgan import DGAN
        from gretel_synthetics.timeseries_dgan.config import DGANConfig
        sample_len = next(s for s in (5, 2, 1) if self.T % s == 0)
        for k in np.unique(y_windows):
            Xk = X_windows[y_windows == k].astype(np.float32)
            torch.manual_seed(self.seed); np.random.seed(self.seed)
            cfg = DGANConfig(max_sequence_len=self.T, sample_len=sample_len, batch_size=min(1000, len(Xk)),
                             epochs=self.epochs, cuda=torch.cuda.is_available(),
                             apply_feature_scaling=True, apply_example_scaling=False)
            m = DGAN(cfg); m.train_numpy(features=Xk)
            self.models[int(k)] = m
            if log:
                log(f"  dgan class {k}: trained on {len(Xk)} windows")
        return self

    def sample_windows(self, labels):
        out = np.zeros((len(labels), self.T, self.d), np.float32)
        for k in np.unique(labels):
            pos = np.where(labels == k)[0]
            _, feats = self.models[int(k)].generate_numpy(len(pos))
            out[pos] = np.clip(np.asarray(feats, dtype=np.float32).reshape(len(pos), self.T, self.d), -1, 1)
        return out
