import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(reg_ts: pd.Series, contamination: float = 0.05, seed: int = 42):
    df = pd.DataFrame({"y": reg_ts}).dropna()
    df["diff1"] = df["y"].diff()
    df["roll6"] = df["y"].rolling(6).mean()
    df["roll12"] = df["y"].rolling(12).mean()
    df["m_sin"], df["m_cos"] = np.sin(2*np.pi*df.index.month/12), np.cos(2*np.pi*df.index.month/12)
    df = df.dropna()
    split = int(len(df) * 0.7)
    tr = df.iloc[:split]
    iso = IsolationForest(contamination=contamination, random_state=seed)
    iso.fit(tr)
    scores = -iso.score_samples(df)
    thr = np.quantile(scores, 0.95)
    anomalies = df.index[scores >= thr]
    return anomalies, scores