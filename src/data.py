"""Load the 10-person, 2-day HNG walking dataset, detect footsteps, build per-step segments and features.

Usage: python -I data.py <raw_dir> <out_pkl>
"""
import os, re, sys, glob, pickle
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import skew, kurtosis

LOCS = ['heel', 'meta', 'toe']
PRE, POST = 0.5, 0.25                       # segment window around the step event (s)
FEATURE_NAMES = ['peak_pos', 'peak_neg', 'peak_to_peak', 'pos_neg_ratio', 'press_to_release_s',
                 'area_pos', 'area_neg', 'rms', 'width_pos_s', 'width_neg_s', 'skewness', 'kurtosis',
                 'zero_crossings', 'step_interval_s', 'band_0_5', 'band_5_15', 'band_15_40', 'spectral_centroid']


def parse_name(n):
    m = re.match(r'Person[-_ ]?(\d+)[-_ ]Day\s*(\d)', n)
    loc = 'heel' if 'heel' in n.lower() else 'meta' if 'meta' in n.lower() else 'toe'
    return int(m.group(1)), int(m.group(2)), loc


def read_file(f):
    raw = pd.read_excel(f, header=None)
    if isinstance(raw.iloc[0, 0], str):
        raw = raw.iloc[1:]
    raw = raw.apply(pd.to_numeric, errors='coerce').dropna()
    return raw[0].values.astype(float), raw[1].values.astype(float)


def preprocess(v, fs, fc=40):
    b, a = butter(4, fc / (fs / 2))
    f = filtfilt(b, a, v)
    return f - np.median(f)


def detect_steps(f, fs):
    w = max(3, int(0.05 * fs))
    env = np.convolve(np.abs(f), np.ones(w) / w, 'same')
    pk, _ = find_peaks(env, distance=int(0.55 * fs), prominence=0.25 * np.percentile(env, 99.5))
    return pk


def segments(f, pk, fs):
    P, Q = int(PRE * fs), int(POST * fs)
    keep, last_end = [], -1
    for p in pk:
        if p - P < 0 or p + Q > len(f):
            continue
        if p - P < last_end:          # would overlap the previous kept segment
            continue
        keep.append(p)
        last_end = p + Q
    keep = np.array(keep, int)
    return keep, np.array([f[p - P:p + Q] for p in keep])


def features(seg, interval, fs):
    s = seg
    mx, mn = s.max(), s.min()
    sp = np.abs(np.fft.rfft(s - s.mean())) ** 2
    fr = np.fft.rfftfreq(len(s), 1 / fs)
    T = sp.sum() + 1e-12
    return np.array([mx, mn, mx - mn, abs(mx / mn) if mn else 0.0, (s.argmin() - s.argmax()) / fs,
                     s[s > 0].sum() / fs, s[s < 0].sum() / fs, np.sqrt((s ** 2).mean()),
                     (s > 0.5 * mx).sum() / fs, (s < 0.5 * mn).sum() / fs, skew(s), kurtosis(s),
                     ((s[:-1] * s[1:]) < 0).sum(), interval,
                     sp[fr < 5].sum() / T, sp[(fr >= 5) & (fr < 15)].sum() / T, sp[(fr >= 15) & (fr < 40)].sum() / T,
                     (fr * sp).sum() / T])


def build(raw_dir):
    recs, rows = {}, []
    for f in sorted(glob.glob(os.path.join(raw_dir, '*.xlsx'))):
        key = parse_name(os.path.basename(f))
        t, v = read_file(f)
        fs = 1 / np.median(np.diff(t))
        fl = preprocess(v, fs)
        pk = detect_steps(fl, fs)
        keep, seg = segments(fl, pk, fs)
        iv = np.diff(pk) / fs
        # interval preceding each kept step (first step: following interval)
        idx = np.searchsorted(pk, keep)
        ints = np.array([iv[i - 1] if i > 0 else iv[0] for i in idx])
        X = np.array([features(s, i, fs) for s, i in zip(seg, ints)])
        clipped = int((np.abs(v) >= 12.65).sum())
        recs[key] = dict(t=t, v=v, f=fl, pk=pk, keep=keep, seg=seg, X=X, fs=fs, clipped=clipped, file=os.path.basename(f))
        rows.append(dict(person=key[0], day=key[1], position=key[2], file=os.path.basename(f), samples=len(v),
                         duration_s=round(t[-1] - t[0], 2), fs_hz=round(fs, 1), steps_detected=len(pk),
                         steps_used=len(keep), clipped_samples=clipped))
    table = pd.DataFrame(rows).sort_values(['person', 'day', 'position']).reset_index(drop=True)
    return recs, table


if __name__ == '__main__':
    recs, table = build(sys.argv[1])
    pickle.dump(dict(recs=recs, table=table, feature_names=FEATURE_NAMES), open(sys.argv[2], 'wb'))
    print(table.to_string(index=False))
    print('recordings', len(table), 'steps used', table.steps_used.sum(), 'persons', table.person.nunique())
