"""Evaluation protocols shared by the classical models and the CNN.

Every protocol yields (name, fold_id, train_keys_and_indices, test_keys_and_indices, label_fn) splits.
A split is a list of (recording_key, step_index_array). Labels are derived from the recording key,
so no information from test recordings is ever used during training.
"""
import numpy as np

LOCS = ['heel', 'meta', 'toe']
K = 5


def lab_person_pos(k):
    return f'P{k[0]}_{k[2]}'


def lab_person(k):
    return f'P{k[0]}'


def lab_pos(k):
    return k[2]


def blocked_groups(n, k=K):
    return np.array_split(np.arange(n), k)


def protocol_A(recs, label_fn, name):
    """Same-day, time-blocked 5-fold CV. Each recording's steps are split into 5 contiguous groups."""
    for day in (1, 2):
        keys = [k for k in recs if k[1] == day]
        for f in range(K):
            tr, te = [], []
            for k in keys:
                g = blocked_groups(len(recs[k]['X']))
                te.append((k, g[f]))
                tr.append((k, np.setdiff1d(np.arange(len(recs[k]['X'])), g[f])))
            yield name, f'day{day}_fold{f}', tr, te, label_fn


def protocol_B(recs, persons=None):
    """Cross-day person identification, one model per position (position assumed known).
    Only persons with that position recorded on both days are included."""
    for loc in LOCS:
        ps = sorted({k[0] for k in recs if k[2] == loc and (k[0], 1, loc) in recs and (k[0], 2, loc) in recs})
        if persons is not None:
            ps = [p for p in ps if p in persons]
        for trd, ted in ((1, 2), (2, 1)):
            tr = [((p, trd, loc), np.arange(len(recs[(p, trd, loc)]['X']))) for p in ps]
            te = [((p, ted, loc), np.arange(len(recs[(p, ted, loc)]['X']))) for p in ps]
            yield 'B_person_crossday', f'{loc}_D{trd}toD{ted}', tr, te, lab_person


def protocol_C(recs):
    """Cross-person position recognition: leave one person out (both days)."""
    for hold in sorted({k[0] for k in recs}):
        tr = [(k, np.arange(len(r['X']))) for k, r in recs.items() if k[0] != hold]
        te = [(k, np.arange(len(r['X']))) for k, r in recs.items() if k[0] == hold]
        yield 'C_position_LOSO', f'P{hold}', tr, te, lab_pos


def gather(recs, split, label_fn, field='X'):
    X = np.vstack([recs[k][field][i] for k, i in split])
    y = np.array(sum([[label_fn(k)] * len(i) for k, i in split], []))
    g = sum([[k] * len(i) for k, i in split], [])
    return X, y, g
