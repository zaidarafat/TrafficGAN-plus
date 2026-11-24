#!/usr/bin/env python
"""Writes tiny fake CSV files with the column layouts of CIC-IDS-2017, CSE-CIC-IDS2018 and UNSW-NB15.
Used only by tests/smoke_test.sh to check that the whole pipeline runs. NOT real data."""
# --- run from the package root, whatever the current directory ---
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
_os.chdir(_ROOT)
if _ROOT not in _sys.path: _sys.path.insert(0, _ROOT)
import os, sys
import numpy as np
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "tests/raw"
rng = np.random.default_rng(0)
C17 = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
       "Fwd Packet Length Max", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Fwd IAT Total",
       "Packet Length Mean", "Min Packet Length", "FIN Flag Count", "Init_Win_bytes_forward", "Active Mean",
       "Idle Mean", "Protocol", "Bwd Avg Bulk Rate"]
C18 = ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "Fwd Pkt Len Max", "Flow Byts/s",
       "Flow Pkts/s", "Flow IAT Mean", "Fwd IAT Tot", "Pkt Len Mean", "Pkt Len Min", "FIN Flag Cnt",
       "Init Fwd Win Byts", "Active Mean", "Idle Mean", "Protocol", "Bwd Blk Rate Avg"]


def flows(n, k, t0):
    base = np.exp(rng.normal(3 + k, 1.0, (n, 1)))
    ar = np.cumsum(rng.normal(0, 0.3, (n, 1)), 0) * 0.05 + 1        # slowly varying, temporally correlated
    X = np.abs(base * ar * rng.lognormal(0, 0.5, (n, len(C17))))
    X[:, 1:5] = np.round(X[:, 1:5]); X[:, 11] = rng.integers(0, 2, n); X[:, 15] = rng.choice([6, 17], n, p=[.8, .2])
    X[:, 16] = 0.0                                                    # constant column (should be dropped)
    X[rng.random(n) < 0.01, 5] = np.inf
    return X


def cic(cols, labels, sizes, path, ts_fmt, extra_id=True):
    rows, labs, ts = [], [], []
    t = pd.Timestamp("2017-07-03 08:00:00")
    for lab, n in zip(labels, sizes):
        k = labels.index(lab)
        X = flows(n, k * 0.7, 0)
        rows.append(X); labs += [lab] * n
    X = np.concatenate(rows); perm = rng.permutation(len(X)); X = X[perm]; labs = np.array(labs)[perm]
    df = pd.DataFrame(X, columns=[" " + c if i % 2 else c for i, c in enumerate(cols)])
    if extra_id:
        df.insert(0, "Flow ID", "1.1.1.1-2.2.2.2"); df.insert(1, " Source IP", "10.0.0.%d" % 1)
        df.insert(2, " Destination Port", rng.integers(1, 65535, len(df)))
    df.insert(3 if extra_id else 0, " Timestamp", [(t + pd.Timedelta(seconds=i)).strftime(ts_fmt) for i in range(len(df))])
    df[" Label"] = labs
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="latin-1")


L17 = ["BENIGN", "DDoS", "PortScan", "DoS Hulk", "FTP-Patator", "SSH-Patator", "Web Attack \x96 Brute Force"]
cic(C17, L17, [6000, 1500, 1200, 1000, 400, 300, 150], f"{root}/cicids2017/Monday-WorkingHours.pcap_ISCX.csv", "%d/%m/%Y %H:%M")
cic(C17, L17, [3000, 800, 600, 500, 200, 200, 50], f"{root}/cicids2017/Tuesday-WorkingHours.pcap_ISCX.csv", "%d/%m/%Y %H:%M")
L18 = ["Benign", "DDOS attack-HOIC", "DoS attacks-Hulk", "FTP-BruteForce", "SSH-Bruteforce", "Bot", "Infilteration"]
cic(C18, L18, [6000, 1500, 1000, 400, 400, 300, 200], f"{root}/cicids2018/02-14-2018.csv", "%d/%m/%Y %H:%M:%S", extra_id=False)

# UNSW-NB15 raw (no header, 49 columns)
from tg.data import UNSW_RAW_COLS
n = 9000
cats = rng.choice(["", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance"], n, p=[.6, .15, .1, .07, .05, .03])
df = pd.DataFrame({c: np.abs(rng.lognormal(2, 1, n)) for c in UNSW_RAW_COLS})
df["srcip"] = "59.166.0.1"; df["dstip"] = "149.171.126.6"; df["proto"] = rng.choice(["tcp", "udp", "arp"], n)
df["state"] = rng.choice(["FIN", "CON", "INT"], n); df["service"] = rng.choice(["-", "http", "dns"], n)
df["Stime"] = 1421927414 + np.arange(n); df["attack_cat"] = cats; df["Label"] = (cats != "").astype(int)
os.makedirs(f"{root}/unsw_nb15", exist_ok=True)
df.to_csv(f"{root}/unsw_nb15/UNSW-NB15_1.csv", index=False, header=False)
print("fake data written to", root)
