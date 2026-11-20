"""1-D CNN under protocols A, B, C with an explicit train / validation / test split (protocol D).

Usage: python -I eval_cnn.py <dataset.pkl> <out_csv> <protocol_name> [aug] [seeds]
Validation data are always taken from the training side of a split:
  A (same day):   one of the four training step-groups of every training recording (contiguous in time)
  B (cross-day):  the last 20 % of steps (contiguous) of every training recording
  C (new person): all recordings of one further training person (the next person id)
Training stops early on validation loss (patience 10, max 100 epochs, best weights restored).
Input: filtered step segments (195 samples) divided by the global SD of the training segments.
"""
import sys, os, pickle, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocols import protocol_A, protocol_B, protocol_C, gather, lab_person_pos, lab_person, blocked_groups, K
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras
tf.config.threading.set_inter_op_parallelism_threads(2)


def build(L, nc):
    m = keras.Sequential([keras.Input((L, 1)),
                          keras.layers.Conv1D(64, 3, activation='relu'), keras.layers.MaxPooling1D(2), keras.layers.Dropout(0.3),
                          keras.layers.Conv1D(128, 3, activation='relu'), keras.layers.MaxPooling1D(2), keras.layers.Flatten(),
                          keras.layers.Dense(128, activation='relu'), keras.layers.Dropout(0.3), keras.layers.Dense(nc, activation='softmax')])
    m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m


def augment(X, rng):
    out = []
    for x in X:
        s = rng.integers(-10, 11)
        x = np.roll(x, s) * rng.uniform(0.8, 1.2) + rng.normal(0, 0.03, len(x))
        out.append(x)
    return np.array(out)


def split_val(name, tr, fold, recs):
    """Return (train_split, val_split) drawn only from the training side."""
    if name.startswith('A_'):
        test_g = int(fold.split('fold')[1])
        vg = (test_g + 1) % K
        trn, val = [], []
        for k, idx in tr:
            g = blocked_groups(len(recs[k]['X']))
            val.append((k, np.intersect1d(idx, g[vg])))
            trn.append((k, np.setdiff1d(idx, g[vg])))
        return trn, val
    if name.startswith('B_'):
        trn, val = [], []
        for k, idx in tr:
            c = int(0.8 * len(idx))
            trn.append((k, idx[:c])); val.append((k, idx[c:]))
        return trn, val
    if name.startswith('C_'):
        held = int(fold[1:])
        persons = sorted({k[0] for k, _ in tr})
        vp = next(p for p in persons if p > held) if any(p > held for p in persons) else persons[0]
        return [(k, i) for k, i in tr if k[0] != vp], [(k, i) for k, i in tr if k[0] == vp]


def main(ds, out, proto, aug=False, seeds=(0, 1, 2)):
    recs = pickle.load(open(ds, 'rb'))['recs']
    subset = os.environ.get('SUBSET', 'all')
    if subset == 'noP1P2':
        recs = {k: v for k, v in recs.items() if k[0] not in (1, 2)}
    gens = {'A_sameday_30class': lambda: protocol_A(recs, lab_person_pos, 'A_sameday_30class'),
            'A_sameday_person': lambda: protocol_A(recs, lab_person, 'A_sameday_person'),
            'B_person_crossday': lambda: protocol_B(recs), 'C_position_LOSO': lambda: protocol_C(recs)}
    rows, curves, preds = [], [], []
    for name, fold, tr, te, lf in gens[proto]():
        trn, val = split_val(name, tr, fold, recs)
        Xtr, ytr, _ = gather(recs, trn, lf, 'seg')
        Xva, yva, _ = gather(recs, val, lf, 'seg')
        Xte, yte, gte = gather(recs, te, lf, 'seg')
        classes = sorted(set(ytr))
        enc = {c: i for i, c in enumerate(classes)}
        sd = Xtr.std()
        for seed in seeds:
            keras.utils.set_random_seed(seed)
            rng = np.random.default_rng(seed)
            Xa = augment(Xtr, rng) if aug else Xtr
            if aug:   # original + one augmented copy
                Xa = np.vstack([Xtr, Xa]); ya = np.concatenate([ytr, ytr])
            else:
                ya = ytr
            m = build(Xtr.shape[1], len(classes))
            va_mask = np.array([y in enc for y in yva])
            es = keras.callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True)
            h = m.fit((Xa / sd)[..., None], np.array([enc[y] for y in ya]), epochs=100, batch_size=32, verbose=0,
                      validation_data=((Xva[va_mask] / sd)[..., None], np.array([enc[y] for y in yva[va_mask]])), callbacks=[es])
            p = np.array(classes)[m.predict((Xte / sd)[..., None], verbose=0).argmax(1)]
            wt, wp = [], []
            for k in dict.fromkeys(gte):
                msk = np.array([g == k for g in gte])
                wt.append(yte[msk][0]); wp.append(Counter(p[msk]).most_common(1)[0][0])
            labels = sorted(set(ytr) | set(yte))
            rows.append(dict(subset=subset, protocol=name, fold=fold, model='CNN_aug' if aug else 'CNN', seed=seed, n_test=len(yte),
                             n_train=len(ytr), n_val=int(va_mask.sum()), epochs=len(h.history['loss']), best_epoch=int(np.argmin(h.history['val_loss'])) + 1,
                             n_classes=len(labels), acc=accuracy_score(yte, p), f1=f1_score(yte, p, average='macro', labels=labels, zero_division=0),
                             walk_acc=float(np.mean(np.array(wt) == np.array(wp))), n_walks=len(wt)))
            curves.append(dict(protocol=name, fold=fold, seed=seed, **{k: v for k, v in h.history.items()}))
            for k, yt, yp in zip(gte, yte, p):
                preds.append((subset, name, fold, rows[-1]['model'], seed, k[0], k[1], k[2], yt, yp))
        print(name, fold, round(np.mean([r['acc'] for r in rows[-len(seeds):]]), 3), flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pickle.dump(curves, open(out.replace('.csv', '_curves.pkl'), 'wb'))
    pd.DataFrame(preds, columns=['subset', 'protocol', 'fold', 'model', 'seed', 'person', 'day', 'position', 'y_true', 'y_pred']
                 ).to_csv(out.replace('.csv', '_preds.csv'), index=False)


if __name__ == '__main__':
    aug = len(sys.argv) > 4 and sys.argv[4] == 'aug'
    main(sys.argv[1], sys.argv[2], sys.argv[3], aug)
