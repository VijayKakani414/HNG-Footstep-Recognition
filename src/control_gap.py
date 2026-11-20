"""Control analysis: how much of the same-day accuracy is a recording 'fingerprint' rather than footstep shape?
Compares RF (manuscript settings) on (a) footstep segments and (b) 0.3 s quiet segments taken midway between
consecutive footsteps (no step inside), both with the same 18 features, under the same-day time-blocked 5-fold CV.
Also reports same-day position recognition (3 classes) for comparison with the cross-person test.
Usage: python -I control_gap.py <dataset.pkl> <out_csv>
"""
import sys, os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import features
from sklearn.ensemble import RandomForestClassifier

d = pickle.load(open(sys.argv[1], 'rb')); R = d['recs']
L = 78   # 0.3 s at 260 Hz
gap = {}
for k, r in R.items():
    f, pk, fs = r['f'], r['pk'], r['fs']; segs = []
    for a, b in zip(pk[:-1], pk[1:]):
        c = (a + b) // 2; s = f[c - L // 2:c - L // 2 + L]
        if len(s) == L:
            segs.append(features(s, (b - a) / fs, fs))
    gap[k] = np.array(segs)
step = {k: r['X'] for k, r in R.items()}


def blocked(Xd, lab, seeds=(0, 1, 2)):
    acc = []
    for seed in seeds:
        for day in (1, 2):
            keys = [k for k in Xd if k[1] == day]
            for f in range(5):
                tr, ytr, te, yte = [], [], [], []
                for k in keys:
                    g = np.array_split(np.arange(len(Xd[k])), 5)[f]; m = np.zeros(len(Xd[k]), bool); m[g] = True
                    tr.append(Xd[k][~m]); ytr += [lab(k)] * (~m).sum(); te.append(Xd[k][m]); yte += [lab(k)] * m.sum()
                mdl = RandomForestClassifier(100, random_state=seed, n_jobs=1).fit(np.vstack(tr), ytr)
                acc.append((mdl.predict(np.vstack(te)) == np.array(yte)).mean())
    return np.mean(acc), np.std(acc)


rows = []
for nm, lab, ch in [('30-class (person x position)', lambda k: f'{k[0]}_{k[2]}', 1 / 30),
                    ('person (10 classes)', lambda k: k[0], 0.1), ('position (3 classes)', lambda k: k[2], 1 / 3)]:
    a, s = blocked(step, lab); b, t = blocked(gap, lab)
    rows.append(dict(task=nm, chance=ch, footstep_acc=a, footstep_sd=s, quiet_gap_acc=b, quiet_gap_sd=t))
    print(f'{nm:30s} footsteps {a:.3f}±{s:.3f} | quiet gaps {b:.3f}±{t:.3f} | chance {ch:.3f}')
pd.DataFrame(rows).to_csv(sys.argv[2], index=False)
