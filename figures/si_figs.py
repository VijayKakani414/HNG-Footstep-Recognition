"""Supplementary figures S9-S12 for the 10-participant analysis.
Usage: python si_figs_new.py <ml10_dir> <out_dir>
"""
import sys, os, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ML, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'], 'font.size': 8,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})
MM = 1 / 25.4
res = lambda f: os.path.join(ML, 'results', f)


def save(fig, name):
    for ext, kw in [('pdf', {}), ('tif', dict(dpi=600, pil_kwargs={'compression': 'tiff_lzw'})), ('png', dict(dpi=300))]:
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), **kw)
    p = os.path.join(OUT, f'{name}.tif'); Image.open(p).convert('RGB').save(p, compression='tiff_lzw', dpi=(600, 600))
    plt.close(fig)


def letter(ax, s):
    ax.text(-0.18, 1.08, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom')


# S9: CNN learning curves (loss), mean +- SD over folds and seeds, NaN after early stop
T = [('A_sameday_30class', 'Same day (30 classes)'), ('A_sameday_person', 'Same day, person (10)'),
     ('B_person_crossday', 'Day 1→2, person (10)'), ('C_position_LOSO', 'New person, position (3)')]
fig, axs = plt.subplots(2, 4, figsize=(174 * MM, 82 * MM), sharex=True)
for j, (p, title) in enumerate(T):
    C = pickle.load(open(res(f'cnn_{p}_curves.pkl'), 'rb')); df = pd.read_csv(res(f'cnn_{p}.csv'))
    E = 40
    for row, (k1, k2, yl) in enumerate([('loss', 'val_loss', 'Loss'), ('accuracy', 'val_accuracy', 'Accuracy')]):
        ax = axs[row, j]
        for key, col, ls, nm in [(k1, '#0072B2', '-', 'Training'), (k2, '#D55E00', '--', 'Validation')]:
            A = np.full((len(C), E), np.nan)
            for i, c in enumerate(C):
                h = c[key][:E]; A[i, :len(h)] = h
            mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
            ep = np.arange(1, E + 1)
            ax.fill_between(ep, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
            ax.plot(ep, mu, c=col, ls=ls, lw=1, label=nm)
        ax.axvline(df.best_epoch.mean(), c='0.4', lw=0.6, ls=':')
        if row == 0:
            ax.set_title(title, fontsize=7.5)
        else:
            ax.set_xlabel('Epoch'); ax.set_ylim(0, 1.02)
        if j == 0:
            ax.set_ylabel(yl)
axs[0, 0].legend(loc='upper right', fontsize=7)
axs[1, 0].text(0.98, 0.04, 'dotted: mean best\nepoch (early stop)', transform=axs[1, 0].transAxes, ha='right', fontsize=6.5)
for ax, l in zip(axs[0], 'abcd'):
    ax.text(-0.25, 1.12, l, transform=ax.transAxes, fontsize=10, fontweight='bold')
plt.tight_layout(); save(fig, 'FigS12_CNN_learning_curves')

# S10: confusion matrices
CM = pickle.load(open(res('classic_all_cms.pkl'), 'rb'))
fig = plt.figure(figsize=(174 * MM, 64 * MM))
a1 = fig.add_axes([0.06, 0.20, 0.27, 0.70]); a2 = fig.add_axes([0.48, 0.20, 0.25, 0.70]); a3 = fig.add_axes([0.81, 0.28, 0.16, 0.55])
k = [k for k in CM if k[0] == 'A_sameday_30class' and k[1] == 'RF'][0]
labs = list(k[2]); order = sorted(range(30), key=lambda i: (int(labs[i].split('_')[0][1:]), labs[i].split('_')[1]))
m = CM[k].astype(float)[np.ix_(order, order)]; m = m / m.sum(1, keepdims=True) * 100
im = a1.imshow(m, cmap='Blues', vmin=0, vmax=100)
a1.set_xticks(np.arange(1, 30, 3), [str(i) for i in range(1, 11)]); a1.set_yticks(np.arange(1, 30, 3), [str(i) for i in range(1, 11)])
for b in np.arange(2.5, 29, 3):
    a1.axhline(b, c='0.85', lw=0.3); a1.axvline(b, c='0.85', lw=0.3)
a1.set_xlabel('Predicted (participant; heel, meta, toe)'); a1.set_ylabel('True'); a1.set_title('Same day, 30 classes (RF)', fontsize=7.5)
letter(a1, 'a')
# cross-day person: merge the three position models into one 10 x 10 matrix
M10 = np.zeros((10, 10))
for kk, v in CM.items():
    if kk[0] == 'B_person_crossday' and kk[1] == 'RF':
        idx = [int(l[1:]) - 1 for l in kk[2]]; M10[np.ix_(idx, idx)] += v
M10 = M10 / M10.sum(1, keepdims=True) * 100
a2.imshow(M10, cmap='Blues', vmin=0, vmax=100)
for i in range(10):
    for j in range(10):
        if M10[i, j] >= 8:
            a2.text(j, i, f'{M10[i, j]:.0f}', ha='center', va='center', fontsize=5.5, color='white' if M10[i, j] > 60 else 'black')
a2.set_xticks(range(10), range(1, 11)); a2.set_yticks(range(10), range(1, 11)); a2.tick_params(labelsize=7, length=0)
a2.set_xlabel('Predicted participant'); a2.set_ylabel('True participant'); a2.set_title('Day 1 → Day 2, person (RF)', fontsize=7.5)
letter(a2, 'b')
k = [k for k in CM if k[0] == 'C_position_LOSO' and k[1] == 'SVM'][0]
m = CM[k].astype(float); m = m / m.sum(1, keepdims=True) * 100
a3.imshow(m, cmap='Blues', vmin=0, vmax=100)
for i in range(3):
    for j in range(3):
        a3.text(j, i, f'{m[i, j]:.0f}', ha='center', va='center', fontsize=7, color='white' if m[i, j] > 60 else 'black')
L3 = ['Heel', 'Meta', 'Toe']; a3.set_xticks(range(3), L3); a3.set_yticks(range(3), L3); a3.tick_params(length=0)
a3.set_xlabel('Predicted'); a3.set_ylabel('True'); a3.set_title('New person, position (SVM)', fontsize=7.5)
letter(a3, 'c')
for a in (a1, a2, a3):
    [s.set_visible(True) for s in a.spines.values()]
cb = fig.colorbar(im, cax=fig.add_axes([0.34, 0.20, 0.01, 0.70])); cb.set_label('Row %', fontsize=7); cb.outline.set_linewidth(0.6)
save(fig, 'FigS11_confusion_matrices')

# S11: recording-fingerprint control
G = pd.read_csv(res('control_gap.csv'))
fig, ax = plt.subplots(figsize=(130 * MM, 60 * MM)); x = np.arange(len(G)); w = 0.34
ax.bar(x - w / 2, G.footstep_acc, w, yerr=G.footstep_sd, color='#08519C', label='Footstep segments', capsize=1.5, error_kw=dict(lw=0.5))
ax.bar(x + w / 2, G.quiet_gap_acc, w, yerr=G.quiet_gap_sd, color='#BDBDBD', hatch='////', edgecolor='white', lw=0,
       label='Quiet segments between footsteps', capsize=1.5, error_kw=dict(lw=0.5))
for i, ch in enumerate(G.chance):
    ax.plot([i - 0.42, i + 0.42], [ch, ch], 'k--', lw=0.7, label='Chance' if i == 0 else None)
ax.set_xticks(x, ['Person × position\n(30 classes)', 'Person\n(10 classes)', 'Position\n(3 classes)'])
ax.set_ylim(0, 1); ax.set_ylabel('Same-day accuracy (RF)'); ax.legend(fontsize=7, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.4, columnspacing=0.8)
plt.tight_layout(); save(fig, 'FigS9_fingerprint_control')

# S12: per-participant cross-day accuracy, all models
PP = pd.read_csv(res('summary_B_perperson.csv'), index_col=0)
order = [m for m in ['KNN', 'SVM', 'RF', 'GB', 'MLP', 'CNN', 'CNN_aug'] if m in PP.index]
PP = PP.loc[order]
fig, ax = plt.subplots(figsize=(120 * MM, 58 * MM))
im = ax.imshow(PP.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i in range(PP.shape[0]):
    for j in range(PP.shape[1]):
        v = PP.values[i, j]; ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5, color='white' if v > 0.6 else 'black')
ax.set_xticks(range(10), range(1, 11)); ax.set_yticks(range(len(order)), [o.replace('CNN_aug', 'CNN + aug.') for o in order])
ax.set_xlabel('Participant'); ax.tick_params(length=0); [s.set_visible(True) for s in ax.spines.values()]
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label('Day 1 → Day 2 accuracy', fontsize=7); cb.outline.set_linewidth(0.6)
plt.tight_layout(); save(fig, 'FigS10_per_participant_crossday')
print('SI figures written')
