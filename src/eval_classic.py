"""Classical models (KNN, SVM, RF, GB, MLP) under protocols A, B, C.

Usage: python -I eval_classic.py <dataset.pkl> <out_csv> [subset]
subset: all | noP1P2 | noclip
Hyperparameters are those of the original manuscript code and are fixed (no tuning).
Writes one row per (protocol, fold, model, seed) with step-level accuracy and macro-F1,
plus walk-level (majority vote per 20 s recording) and, for protocol B, day-level votes.
"""
import sys, os, pickle, warnings
import numpy as np, pandas as pd
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocols import protocol_A, protocol_B, protocol_C, gather, lab_person_pos, lab_person
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEEDED = {'RF', 'MLP'}


def models(seed):
    return {'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
            'SVM': make_pipeline(StandardScaler(), SVC()),
            'RF': RandomForestClassifier(100, random_state=seed, n_jobs=-1),
            'GB': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed),
            'MLP': make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=1000, random_state=seed))}


def vote(labels):
    return Counter(labels).most_common(1)[0][0]


def main(ds, out, subset='all', seeds=(0, 1, 2), only=None):
    d = pickle.load(open(ds, 'rb'))
    recs = d['recs']
    if subset == 'noP1P2':
        recs = {k: v for k, v in recs.items() if k[0] not in (1, 2)}
    if subset == 'noclip':
        recs = {k: v for k, v in recs.items() if v['clipped'] == 0}
    splits = list(protocol_A(recs, lab_person_pos, 'A_sameday_30class')) + \
             list(protocol_A(recs, lab_person, 'A_sameday_person')) + \
             list(protocol_B(recs)) + list(protocol_C(recs))
    if only:
        splits = [s for s in splits if s[0] in only]
    rows, cms, preds = [], defaultdict(lambda: None), []
    for name, fold, tr, te, lf in splits:
        Xtr, ytr, _ = gather(recs, tr, lf)
        Xte, yte, gte = gather(recs, te, lf)
        labels = sorted(set(ytr) | set(yte))
        for seed in seeds:
            for m, mod in models(seed).items():
                if seed != seeds[0] and m not in SEEDED:
                    continue
                mod.fit(Xtr, ytr)
                p = mod.predict(Xte)
                # walk-level vote (one 20 s recording)
                wt, wp = [], []
                for k in dict.fromkeys(gte):
                    msk = np.array([g == k for g in gte])
                    wt.append(yte[msk][0]); wp.append(vote(p[msk]))
                rows.append(dict(subset=subset, protocol=name, fold=fold, model=m, seed=seed, n_test=len(yte),
                                 n_classes=len(labels), acc=accuracy_score(yte, p),
                                 f1=f1_score(yte, p, average='macro', labels=labels, zero_division=0),
                                 walk_acc=float(np.mean(np.array(wt) == np.array(wp))), n_walks=len(wt)))
                for k, yt, yp in zip(gte, yte, p):
                    preds.append((subset, name, fold, m, seed, k[0], k[1], k[2], yt, yp))
                key = (name, m, tuple(labels))
                cm = confusion_matrix(yte, p, labels=labels)
                cms[key] = cm if cms[key] is None else cms[key] + cm
        print(name, fold, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    pd.DataFrame(preds, columns=['subset', 'protocol', 'fold', 'model', 'seed', 'person', 'day', 'position', 'y_true', 'y_pred']
                 ).to_csv(out.replace('.csv', '_preds.csv'), index=False)
    pickle.dump(dict(cms), open(out.replace('.csv', '_cms.pkl'), 'wb'))
    return df


if __name__ == '__main__':
    subset = sys.argv[3] if len(sys.argv) > 3 else 'all'
    only = sys.argv[4].split(',') if len(sys.argv) > 4 else None
    df = main(sys.argv[1], sys.argv[2], subset, only=only)
    print(df.groupby(['protocol', 'model']).agg(acc=('acc', 'mean'), sd=('acc', 'std'), f1=('f1', 'mean'),
                                               walk=('walk_acc', 'mean')).round(3).to_string())
