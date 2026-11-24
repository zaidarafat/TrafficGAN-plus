"""Loading, cleaning, windowing, splitting and scaling of flow datasets.

Supported datasets (put the original CSV files under data/raw/<name>/):
  cicids2017  CIC-IDS-2017 "GeneratedLabelledFlows"/"TrafficLabelling" or "MachineLearningCVE" CSVs
  cicids2018  CSE-CIC-IDS2018 "Processed Traffic Data for ML Algorithms" CSVs
  unsw_nb15   UNSW-NB15 raw CSVs (UNSW-NB15_1..4.csv, no header) or the
              UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv files (with header)

Pipeline (Section 3.7 of the manuscript):
  1. read files in chronological order, chunk by chunk;
  2. per class, cut consecutive flows of that class into non-overlapping windows of T flows;
  3. keep at most `max_windows_per_class_total` windows per class by uniform random sampling;
  4. select benign + the (k_max - 1) largest attack classes with >= min_flows_per_class flows;
  5. stratified 70/15/15 split at window level (no flow can occur in two partitions);
  6. drop identifier columns and columns that are constant on the training partition,
     encode categorical strings, replace +-inf by the column's maximum finite value, NaN by 0;
  7. optional signed log1p on continuous columns, then min-max scaling to [-1, 1] fitted on
     the training partition only.
"""
from __future__ import annotations
import glob, os, re, pickle, json
import numpy as np
import pandas as pd

UNSW_RAW_COLS = ["srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
    "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin",
    "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime",
    "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "Label"]

SPEC = {
    "cicids2017": dict(label="Label", order="row", time=None,
        drop=["Flow ID", "Source IP", "Src IP", "Source Port", "Src Port", "Destination IP", "Dst IP",
              "Destination Port", "Dst Port", "Timestamp", "Fwd Header Length.1"],
        categorical=[]),
    "cicids2018": dict(label="Label", order="timestamp", time="Timestamp", time_format="%d/%m/%Y %H:%M:%S",
        drop=["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp"],
        categorical=[]),
    "unsw_nb15": dict(label="attack_cat", order="stime", time="Stime",
        drop=["srcip", "sport", "dstip", "dsport", "Stime", "Ltime", "id", "label", "Label", "stime", "ltime"],
        categorical=["proto", "service", "state"]),
}
BENIGN_NAMES = {"benign", "normal"}
DAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TIMING_RE = re.compile(r"(duration|\biat\b|iat|active|idle|^dur$|jit|intpkt|tcprtt|synack|ackdat)", re.I)


def canon_label(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^\x20-\x7E]+", "-", s)           # 'Web Attack \x96 Brute Force' -> 'Web Attack - Brute Force'
    s = re.sub(r"\s+", " ", s).strip()
    if s == "" or s.lower() in ("nan", "none"):
        s = "Normal"
    if s == "Backdoors":
        s = "Backdoor"
    if s.lower() in BENIGN_NAMES:
        return "Benign"
    return s


def _file_sort_key(path: str):
    name = os.path.basename(path).lower()
    day = next((i for i, d in enumerate(DAY_ORDER) if d in name), 99)
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})", name)            # CSE-CIC-IDS2018 file names carry dates
    date = (int(m.group(3)), int(m.group(2)), int(m.group(1))) if m else (0, 0, 0)
    part = 0 if "morning" in name else 1 if "afternoon" in name else 0
    num = re.findall(r"_(\d+)\.csv$", name)
    return (date, day, part, int(num[0]) if num else 0, name)


def _read_chunks(path: str, dataset: str, chunksize: int):
    if dataset == "unsw_nb15":
        with open(path, "r", errors="ignore") as fh:
            first = fh.readline()
        has_header = "attack_cat" in first
        kw = dict(header=0) if has_header else dict(header=None, names=UNSW_RAW_COLS)
        return pd.read_csv(path, chunksize=chunksize, low_memory=False, encoding="latin-1", **kw)
    return pd.read_csv(path, chunksize=chunksize, low_memory=False, encoding="latin-1",
                       skipinitialspace=True)


def load_windows(dataset: str, raw_dir: str, T: int, max_windows: int, seed: int = 0,
                 chunksize: int = 500_000, verbose: bool = True):
    """Read all CSV files, build per-class non-overlapping windows, uniformly subsample windows per class.

    Returns (frames, total_windows, total_flows) where frames[label] is a DataFrame with
    T * n_windows rows (window i occupies rows i*T ... i*T+T-1)."""
    spec = SPEC[dataset]
    files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True), key=_file_sort_key)
    if dataset == "unsw_nb15":
        raw = [f for f in files if re.search(r"UNSW-NB15_\d\.csv$", f)]
        files = raw if raw else [f for f in files if re.search(r"(training|testing)-set", f, re.I)]
    files = [f for f in files if "feature" not in os.path.basename(f).lower()
             and "gt" != os.path.basename(f).lower()[:2]]
    if not files:
        raise FileNotFoundError(f"no CSV files for {dataset} under '{raw_dir}'. Set paths.datasets.{dataset} in "
                                f"configs/default.yaml or pass --data-dir \"<folder with the CSV files>\".")
    rng = np.random.default_rng(seed)
    kept: dict[str, pd.DataFrame] = {}
    keys: dict[str, np.ndarray] = {}
    total_windows: dict[str, int] = {}
    total_flows: dict[str, int] = {}
    carry: dict[str, pd.DataFrame] = {}      # incomplete window tail, continued in the next chunk of the same file
    for path in files:
        if verbose:
            print(f"  reading {os.path.basename(path)}", flush=True)
        carry = {}
        for chunk in _read_chunks(path, dataset, chunksize):
            chunk.columns = [str(c).strip() for c in chunk.columns]
            lab = spec["label"]
            if lab not in chunk.columns and "Label" in chunk.columns:
                lab = "Label"
            chunk = chunk[chunk[lab].astype(str).str.strip() != lab]            # repeated header rows (2018)
            chunk["_label"] = chunk[lab].map(canon_label)
            if spec["order"] == "timestamp" and spec["time"] in chunk.columns:
                ts = pd.to_datetime(chunk[spec["time"]], format=spec.get("time_format"), errors="coerce")
                chunk = chunk.assign(_t=ts.astype("int64")).sort_values("_t", kind="stable").drop(columns="_t")
            elif spec["order"] == "stime" and spec["time"] in chunk.columns:
                chunk = chunk.sort_values(spec["time"], kind="stable")
            for label, g in chunk.groupby("_label", sort=False):
                total_flows[label] = total_flows.get(label, 0) + len(g)
                if label in carry:
                    g = pd.concat([carry.pop(label), g])
                n = len(g) // T
                if len(g) % T:
                    carry[label] = g.iloc[n * T:]
                if n == 0:
                    continue
                g = g.iloc[: n * T]
                total_windows[label] = total_windows.get(label, 0) + n
                k = rng.random(n)
                if label in kept:
                    k_all = np.concatenate([keys[label], k])
                    df_all = pd.concat([kept[label], g], ignore_index=True)
                else:
                    k_all, df_all = k, g.reset_index(drop=True)
                if len(k_all) > max_windows:
                    sel = np.sort(np.argpartition(k_all, max_windows)[:max_windows])
                    rows = (sel[:, None] * T + np.arange(T)[None, :]).ravel()
                    df_all = df_all.iloc[rows].reset_index(drop=True)
                    k_all = k_all[sel]
                kept[label], keys[label] = df_all, k_all
    return kept, total_windows, total_flows


class Preprocessor:
    """Column handling, signed-log transform and [-1, 1] min-max scaling (fitted on training flows)."""

    def fit(self, df: pd.DataFrame, feature_cols, categorical, discrete_max_unique, log_transform):
        self.feature_cols = list(feature_cols)
        self.cat_maps = {}
        for c in categorical:
            if c in self.feature_cols:
                vc = df[c].astype(str).value_counts()
                self.cat_maps[c] = {v: i for i, v in enumerate(vc.index)}   # frequency-rank codes
        X = self._numeric(df)
        keep = np.nanmax(X, 0) > np.nanmin(X, 0)                             # drop constant columns
        self.feature_cols = [c for c, k in zip(self.feature_cols, keep) if k]
        X = X[:, keep]
        self.colmax, self.colmin = self.colmax[keep], self.colmin[keep]
        self.discrete = np.array([(c in self.cat_maps) or
                                  (np.all(np.mod(X[:, j], 1) == 0) and len(np.unique(X[:, j])) <= discrete_max_unique)
                                  for j, c in enumerate(self.feature_cols)])
        self.log_mask = (~self.discrete) & bool(log_transform)
        self.discrete_values = {j: np.unique(X[:, j]) for j in np.where(self.discrete)[0]}
        Z = self._fwd(X)
        self.lo, self.hi = Z.min(0), Z.max(0)
        self.timing_idx = [j for j, c in enumerate(self.feature_cols) if TIMING_RE.search(c)]
        return self

    def _raw(self, df):
        cols = []
        for c in self.feature_cols:
            if c in self.cat_maps:
                cols.append(df[c].astype(str).map(self.cat_maps[c]).fillna(len(self.cat_maps[c])).to_numpy(np.float64))
            else:
                cols.append(pd.to_numeric(df[c], errors="coerce").to_numpy(np.float64))
        return np.stack(cols, 1)

    def _numeric(self, df):
        """+inf -> column maximum finite value, -inf -> column minimum finite value, NaN -> 0."""
        X = self._raw(df)
        if not hasattr(self, "colmax"):
            fin = np.isfinite(X)
            self.colmax = np.where(fin.any(0), np.max(np.where(fin, X, -np.inf), 0), 0.0)
            self.colmin = np.where(fin.any(0), np.min(np.where(fin, X, np.inf), 0), 0.0)
        X = np.where(np.isposinf(X), self.colmax[None, :], X)
        X = np.where(np.isneginf(X), self.colmin[None, :], X)
        return np.nan_to_num(X, nan=0.0)

    def _fwd(self, X):
        Z = X.copy()
        Z[:, self.log_mask] = np.sign(Z[:, self.log_mask]) * np.log1p(np.abs(Z[:, self.log_mask]))
        return Z

    def transform_df(self, df):
        X = self._numeric(df)
        return self.scale(X)

    def scale(self, X):
        Z = self._fwd(X)
        rng = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        return np.clip(2 * (Z - self.lo) / rng - 1, -1.5, 1.5).astype(np.float32)

    def inverse(self, S):
        """Scaled [-1,1] values (..., d) -> original feature space; discrete columns snapped to observed values."""
        shp = S.shape
        S = S.reshape(-1, shp[-1]).astype(np.float64)
        Z = (np.clip(S, -1, 1) + 1) / 2 * (self.hi - self.lo) + self.lo
        X = Z.copy()
        X[:, self.log_mask] = np.sign(Z[:, self.log_mask]) * np.expm1(np.abs(Z[:, self.log_mask]))
        for j, vals in self.discrete_values.items():
            idx = np.clip(np.searchsorted(vals, X[:, j]), 1, len(vals) - 1)
            left, right = vals[idx - 1], vals[idx]
            X[:, j] = np.where(np.abs(X[:, j] - left) <= np.abs(right - X[:, j]), left, right)
        return X.reshape(shp)

    def snap(self, S):
        """Round generated windows to valid discrete values and re-scale (used before evaluation)."""
        return self.scale(self.inverse(S).reshape(-1, S.shape[-1])).reshape(S.shape)


def prepare(dataset: str, cfg: dict, out_dir: str, T: int | None = None, verbose=True):
    dcfg = cfg["data"]
    T = T or dcfg["T"]
    raw_dir = (cfg["paths"].get("datasets") or {}).get(dataset) or os.path.join(cfg["paths"]["raw_root"], dataset)
    spec = SPEC[dataset]
    max_total = int(dcfg.get("max_windows_per_class_total") or 40000)
    frames, tot_w, tot_f = load_windows(dataset, raw_dir, T, max_total, seed=dcfg["split_seed"], verbose=verbose)
    # ---- class selection ----
    counts = {k: tot_w[k] * T for k in frames}
    attacks = sorted([k for k in frames if k != "Benign" and counts[k] >= dcfg["min_flows_per_class"]],
                     key=lambda k: -counts[k])
    classes = (["Benign"] if "Benign" in frames else []) + attacks[: dcfg["k_max"] - 1]
    if verbose:
        print("  classes:", classes)
    # ---- stratified window-level split ----
    rng = np.random.default_rng(dcfg["split_seed"])
    parts = {"train": [], "val": [], "test": []}
    for ci, c in enumerate(classes):
        df = frames[c]
        n = len(df) // T
        perm = rng.permutation(n)
        n_tr, n_va = int(round(dcfg["split"][0] * n)), int(round(dcfg["split"][1] * n))
        idx = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
        caps = {"train": dcfg.get("max_train_windows_per_class"), "val": dcfg.get("max_eval_windows_per_class"),
                "test": dcfg.get("max_eval_windows_per_class")}
        for p in parts:
            w = np.sort(idx[p][: caps[p]] if caps[p] else idx[p])
            rows = (w[:, None] * T + np.arange(T)[None, :]).ravel()
            parts[p].append((df.iloc[rows], np.full(len(w), ci)))
    # ---- features ----
    all_cols = [c for c in frames[classes[0]].columns]
    drop = set(spec["drop"]) | {spec["label"], "Label", "_label", "attack_cat"}
    feature_cols = [c for c in all_cols if c not in drop and not c.startswith("_")]
    train_df = pd.concat([d for d, _ in parts["train"]])
    if dcfg.get("dedup"):
        print("  note: dedup=true only drops duplicate *windows*; flow-level dedup would break windows")
    prep = Preprocessor().fit(train_df, feature_cols, spec["categorical"], dcfg["discrete_max_unique"],
                              dcfg["log_transform"])
    prep.classes, prep.T, prep.dataset = classes, T, dataset
    out = {}
    for p in parts:
        X = np.concatenate([prep.transform_df(d) for d, _ in parts[p]]).reshape(-1, T, len(prep.feature_cols))
        y = np.concatenate([l for _, l in parts[p]])
        out[f"X_{p}"], out[f"y_{p}"] = X, y.astype(np.int64)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{dataset}_T{T}"
    np.savez_compressed(os.path.join(out_dir, tag + ".npz"), **out)
    with open(os.path.join(out_dir, tag + "_prep.pkl"), "wb") as fh:
        pickle.dump(prep, fh)
    summary = {"dataset": dataset, "T": T, "d": len(prep.feature_cols), "classes": classes,
               "features": prep.feature_cols, "discrete": [c for c, m in zip(prep.feature_cols, prep.discrete) if m],
               "timing_features": [prep.feature_cols[j] for j in prep.timing_idx],
               "flows_in_files": {k: int(tot_f[k]) for k in tot_f},
               "windows_in_files": {k: int(tot_w[k]) for k in tot_w},
               "windows_per_split": {p: {c: int((out[f"y_{p}"] == i).sum()) for i, c in enumerate(classes)} for p in parts}}
    with open(os.path.join(out_dir, tag + "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    if verbose:
        print(json.dumps(summary["windows_per_split"], indent=1))
    return out, prep


def load_prepared(dataset: str, cfg: dict, T: int | None = None):
    T = T or cfg["data"]["T"]
    d = cfg["paths"]["processed_root"]
    tag = f"{dataset}_T{T}"
    z = np.load(os.path.join(d, tag + ".npz"))
    with open(os.path.join(d, tag + "_prep.pkl"), "rb") as fh:
        prep = pickle.load(fh)
    return {k: z[k] for k in z.files}, prep
