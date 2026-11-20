"""Recording-level label-permutation test.
Training labels are permuted across whole training recordings (so within-recording dependence is kept);
test labels stay fixed. p = (1 + #null >= observed) / (1 + n_perm).
Usage: python -I permtest.py <dataset.pkl> <protocol> <model> <n_perm> <out_csv>
"""
import sys, os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocols import protocol_A, protocol_B, protocol_C, gather, lab_person_pos, lab_person
from eval_classic import models


def run(recs, splits, model, perm_rng=None):
    correct = n = 0
    for name, fold, tr, te, lf in splits:
        if perm_rng is not None:
            keys = [k for k, _ in tr]
            labs = [lf(k) for k in keys]
            perm = perm_rng.permutation(len(keys))
            lmap = {keys[i]: labs[perm[i]] for i in range(len(keys))}
            ytr = np.array(sum([[lmap[k]] * len(i) for k, i in tr], []))
            Xtr = np.vstack([recs[k]['X'][i] for k, i in tr])
        else:
            Xtr, ytr, _ = gather(recs, tr, lf)
        Xte, yte, _ = gather(recs, te, lf)
        m = models(0)[model]; m.fit(Xtr, ytr)
        correct += (m.predict(Xte) == yte).sum(); n += len(yte)
    return correct / n


if __name__ == '__main__':
    ds, proto, model, nperm, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    recs = pickle.load(open(ds, 'rb'))['recs']
    if os.environ.get('SUBSET') == 'noP1P2':
        recs = {k: v for k, v in recs.items() if k[0] not in (1, 2)}
    gen = {'A_sameday_30class': lambda: protocol_A(recs, lab_person_pos, 'A_sameday_30class'),
           'A_sameday_person': lambda: protocol_A(recs, lab_person, 'A_sameday_person'),
           'B_person_crossday': lambda: protocol_B(recs), 'C_position_LOSO': lambda: protocol_C(recs)}[proto]
    splits = list(gen())
    obs = run(recs, splits, model)
    rng = np.random.default_rng(0)
    null = np.array([run(recs, splits, model, rng) for _ in range(nperm)])
    p = (1 + (null >= obs).sum()) / (1 + nperm)
    pd.DataFrame([dict(subset=os.environ.get('SUBSET', 'all'), protocol=proto, model=model, observed=obs, null_mean=null.mean(), null_max=null.max(), n_perm=nperm, p=p)]).to_csv(out, index=False)
    print(proto, model, f'obs {obs:.3f} null mean {null.mean():.3f} max {null.max():.3f} p={p:.4f}')
