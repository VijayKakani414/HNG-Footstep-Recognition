"""Improvement attempts (all reported, none selected on test data).

I1  nested-CV tuning of SVM and RF: hyperparameters chosen by an inner CV on the training side only
    (A: time-blocked inner folds; B: time-blocked inner folds of the training day; C: leave-one-person-out
    among the training persons).
I4  per person-day feature calibration for protocol C (z-score each person-day's steps using that
    person-day's own statistics, pooled over positions; labels are not used).
Usage: python -I eval_nested.py <dataset.pkl> <out_csv> <which>   which = I1_A | I1_B | I1_C | I4_C
"""
import sys, os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocols import protocol_A, protocol_B, protocol_C, gather, lab_person_pos, blocked_groups, K
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score

GRIDS = {'SVM': (lambda: make_pipeline(StandardScaler(), SVC()),
                 {'svc__C': [0.3, 1, 3, 10, 30, 100], 'svc__gamma': ['scale', 0.01, 0.03, 0.1, 0.3]}),
         'RF': (lambda: RandomForestClassifier(random_state=0, n_jobs=-1),
                {'n_estimators': [100, 300], 'max_depth': [None, 10, 5], 'max_features': ['sqrt', 0.5]})}


def inner_folds(name, tr, recs):
    """Return a fold id per training row, never mixing information across the outer test set."""
    if name.startswith('C_'):
        ps = sorted({k[0] for k, _ in tr})
        return np.concatenate([[ps.index(k[0]) % 4] * len(i) for k, i in tr])
    # A and B: contiguous time blocks inside each training recording
    out = []
    for k, idx in tr:
        g = np.array_split(np.arange(len(idx)), 3)
        f = np.zeros(len(idx), int)
        for j, gg in enumerate(g):
            f[gg] = j
        out.append(f)
    return np.concatenate(out)


def calibrate(recs):
    out = {}
    for p in {k[0] for k in recs}:
        for d in (1, 2):
            ks = [k for k in recs if k[0] == p and k[1] == d]
            if not ks:
                continue
            A = np.vstack([recs[k]['X'] for k in ks]); mu, sd = A.mean(0), A.std(0) + 1e-9
            for k in ks:
                out[k] = dict(recs[k], X=(recs[k]['X'] - mu) / sd)
    return out


def main(ds, out, which):
    recs = pickle.load(open(ds, 'rb'))['recs']
    if which == 'I1_A':
        splits = list(protocol_A(recs, lab_person_pos, 'A_sameday_30class'))
    elif which == 'I1_B':
        splits = list(protocol_B(recs))
    elif which == 'I1_C':
        splits = list(protocol_C(recs))
    elif which == 'I4_C':
        recs = calibrate(recs); splits = list(protocol_C(recs))
    rows = []
    for name, fold, tr, te, lf in splits:
        Xtr, ytr, _ = gather(recs, tr, lf); Xte, yte, _ = gather(recs, te, lf)
        labels = sorted(set(ytr) | set(yte))
        for m, (mk, grid) in GRIDS.items():
            if which.startswith('I4'):
                est = mk(); est.fit(Xtr, ytr); best = 'default'
            else:
                gs = GridSearchCV(mk(), grid, cv=PredefinedSplit(inner_folds(name, tr, recs)), n_jobs=1)
                gs.fit(Xtr, ytr); est = gs.best_estimator_; best = str(gs.best_params_)
            p = est.predict(Xte)
            rows.append(dict(attempt=which, protocol=name, fold=fold, model=m, best_params=best, n_test=len(yte),
                             acc=accuracy_score(yte, p), f1=f1_score(yte, p, average='macro', labels=labels, zero_division=0)))
        print(which, fold, [round(r['acc'], 3) for r in rows[-2:]], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
