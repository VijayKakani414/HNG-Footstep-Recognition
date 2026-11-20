"""Fig. 8 for the revised manuscript (10 participants, 2 days).
Usage: python fig8_new.py <ml10_dir> <panel_a.png or ''> <out_dir>
Springer artwork: 174 mm wide, <= 234 mm high, sans-serif 8 pt, panel letters 10 pt bold, RGB.
"""
import sys, os, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, FancyArrowPatch
from PIL import Image

ML, PA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans'], 'font.size': 8,
                     'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'lines.linewidth': 0.8, 'hatch.linewidth': 0.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False})
MM = 1 / 25.4; FW, FH = 174, 234
R = pickle.load(open(os.path.join(ML, 'results/dataset.pkl'), 'rb'))['recs']
S = pd.read_csv(os.path.join(ML, 'results/summary_main.csv'))
S = S[S.subset == 'all']
CM = pickle.load(open(os.path.join(ML, 'results/classic_all_cms.pkl'), 'rb'))
PP = pd.read_csv(os.path.join(ML, 'results/summary_B_perperson.csv'), index_col=0)
fig = plt.figure(figsize=(FW * MM, FH * MM))


def ax_mm(x, y, w, h, **kw):
    return fig.add_axes([x / FW, 1 - (y + h) / FH, w / FW, h / FH], **kw)


def letter(x, y, s):
    fig.text(x / FW, 1 - y / FH, s, fontsize=10, fontweight='bold', va='top', ha='left')


LOC = {'heel': ('Heel', '#0072B2'), 'meta': ('Meta', '#D55E00'), 'toe': ('Toe', '#009E73')}

# a: original schematic (super-resolved)
if PA and os.path.exists(PA):
    pa = np.array(Image.open(PA).convert('RGB'))
    aw = 120; ah = aw * pa.shape[0] / pa.shape[1]
    a = ax_mm((FW - aw) / 2 + 2, 2, aw, ah); a.imshow(pa, interpolation='none'); a.set_axis_off()
else:
    a = ax_mm(27, 2, 120, 56); a.set_axis_off(); a.text(.5, .5, 'panel a', ha='center', transform=a.transAxes)
letter(1, 2, 'a')

# b, c: example participant, Day 1 and Day 2
P = 5
for day, x0, lab in [(1, 14, 'b'), (2, 101, 'c')]:
    for i, l in enumerate(['heel', 'meta', 'toe']):
        r = R[(P, day, l)]; name, col = LOC[l]
        ax = ax_mm(x0, 66 + i * 10.5, 70, 8.6)
        ax.plot(r['t'] - r['t'][0], r['v'], lw=0.45, c=col)
        ax.set_xlim(0, 20); ax.set_ylabel(name, fontsize=7.5, rotation=0, ha='right', va='center', labelpad=2)
        m = np.ceil(np.abs(r['v']).max()); ax.set_yticks([-m, 0, m]); ax.set_ylim(-m * 1.08, m * 1.08)
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)')
        if i == 0:
            ax.text(0, 1.12, f'Participant {P}, Day {day}: voltage (V)', transform=ax.transAxes, fontsize=8, va='bottom')
    letter(x0 - 13 if day == 1 else x0 - 13, 61, lab)

# d: footstep detection and segmentation
r = R[(P, 1, 'heel')]; t = r['t'] - r['t'][0]; fs = r['fs']
d1 = ax_mm(14, 110, 70, 11); d2 = ax_mm(14, 124, 70, 13)
d1.plot(t, r['v'], lw=0.45, c='#555'); d1.set_xlim(0, 8); d1.set_xticklabels([]); d1.set_ylabel('Raw (V)')
for p in r['keep']:
    d2.axvspan((p - int(0.5 * fs)) / fs, (p + int(0.25 * fs)) / fs, color='#FBE0C4', lw=0, zorder=0)
d2.plot(t, r['f'], lw=0.7, c='#0072B2', zorder=2)
d2.plot(r['keep'] / fs, r['f'][r['keep']], 'v', ms=3, mfc='#D55E00', mec='#D55E00', zorder=3)
skip = np.setdiff1d(r['pk'], r['keep'])
d2.plot(skip / fs, r['f'][skip], 'v', ms=3, mfc='white', mec='#D55E00', mew=0.6, zorder=3)
d2.set_xlim(0, 8); d2.set_xlabel('Time (s)'); d2.set_ylabel('Filtered (V)')
d1.legend(handles=[Patch(fc='#FBE0C4', label='Segment (0.75 s)'),
                   plt.Line2D([], [], ls='', marker='v', ms=3, c='#D55E00', label='Step used'),
                   plt.Line2D([], [], ls='', marker='v', ms=3, mfc='white', mec='#D55E00', label='Not used (overlap)')],
          ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), handlelength=1.0, columnspacing=0.6, fontsize=7)
letter(1, 104, 'd')

# e: the three evaluation protocols
e = ax_mm(100, 106, 72, 34); e.set_xlim(0, 100); e.set_ylim(0, 36); e.set_axis_off()
TR, TE = '#CFE3F2', '#0072B2'
e.text(0, 34.5, 'Same day (time-blocked 5-fold)', fontsize=7.5, va='top')
for j in range(5):
    e.add_patch(Rectangle((2 + j * 9, 26.5), 8.4, 4, fc=TE if j == 2 else TR, ec='white', lw=0.5))
e.text(49, 28.5, 'one 20 s walk', fontsize=7, va='center')
e.text(0, 22.5, 'Day 1 → Day 2 (person, 10 classes)', fontsize=7.5, va='top')
e.add_patch(Rectangle((2, 14.5), 24, 4, fc=TR, ec='none')); e.text(14, 16.5, 'Day 1', ha='center', va='center', fontsize=7)
e.add_patch(Rectangle((34, 14.5), 24, 4, fc=TE, ec='none')); e.text(46, 16.5, 'Day 2', ha='center', va='center', fontsize=7, color='white')
e.annotate('', xy=(33.5, 16.5), xytext=(26.5, 16.5), arrowprops=dict(arrowstyle='->', lw=0.6))
e.text(62, 16.5, '(and reverse)', fontsize=7, va='center')
e.text(0, 10.5, 'New person (position, leave-one-out)', fontsize=7.5, va='top')
for j in range(10):
    e.add_patch(Rectangle((2 + j * 5.6, 2.5), 5.0, 4, fc=TE if j == 6 else TR, ec='none'))
e.text(60, 4.5, '9 train / 1 test', fontsize=7, va='center')
e.legend(handles=[Patch(fc=TR, label='Train'), Patch(fc=TE, label='Test')], loc='upper right', bbox_to_anchor=(1.0, 1.02), ncol=2,
         handlelength=1.2, columnspacing=0.8)
letter(88, 104, 'e')

# f: accuracy of six models in the three tests
M = ['KNN', 'SVM', 'RF', 'GB', 'MLP', 'CNN']
G = [('A_sameday_30class', 'Same day\nperson × position (30)'), ('A_sameday_person', 'Same day\nperson (10)'),
     ('B_person_crossday', 'Day 1 → Day 2\nperson (10)'), ('C_position_LOSO', 'New person\nposition (3)')]
COLS = ['#BDD7EE', '#6BAED6', '#08519C', '#9E9AC8', '#FDAE6B', '#E6550D']
HAT = [None, None, '////', None, None, '....']
ax = ax_mm(14, 154, 156, 26); w = 0.13
for i, m in enumerate(M):
    vals = [S[(S.protocol == g) & (S.model == m)].acc.values[0] for g, _ in G]
    sds = [S[(S.protocol == g) & (S.model == m)].sd.values[0] for g, _ in G]
    ax.bar(np.arange(4) + (i - 2.5) * w, vals, w, yerr=sds, color=COLS[i], hatch=HAT[i], edgecolor='white' if HAT[i] else COLS[i], lw=0,
           label=m, capsize=1.2, error_kw=dict(lw=0.5, capthick=0.5))
for k, (g, _) in enumerate(G):
    ch = S[S.protocol == g].chance.values[0]
    ax.plot([k - 0.45, k + 0.45], [ch, ch], ls='--', c='k', lw=0.7, label='Chance' if k == 0 else None)
ax.set_xticks(range(4), [l for _, l in G]); ax.set_ylim(0, 1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel('Accuracy per footstep'); ax.tick_params(axis='x', length=0)
ax.legend(ncol=7, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.3, columnspacing=1.0)
letter(1, 147, 'f')

# g: same-day person confusion matrix (RF, pooled over folds, days and seeds), row-normalised
key = [k for k in CM if k[0] == 'A_sameday_person' and k[1] == 'RF'][0]
labs = list(key[2]); cm = CM[key].astype(float); cm = cm / cm.sum(1, keepdims=True) * 100
order = sorted(range(len(labs)), key=lambda i: int(labs[i][1:]))
cm = cm[np.ix_(order, order)]; L = [labs[i][1:] for i in order]
ax = ax_mm(22, 194, 29, 29); im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=100)
for i in range(10):
    for j in range(10):
        if cm[i, j] >= 5:
            ax.text(j, i, f'{cm[i, j]:.0f}', ha='center', va='center', fontsize=5.5, color='white' if cm[i, j] > 60 else 'black')
ax.set_xticks(range(10), L); ax.set_yticks(range(10), L); ax.tick_params(labelsize=6.5, length=0)
ax.set_xlabel('Predicted participant'); ax.set_ylabel('True participant'); [s.set_visible(True) for s in ax.spines.values()]
cb = fig.colorbar(im, cax=ax_mm(53, 194, 1.8, 29)); cb.set_label('Footsteps (%)', fontsize=7); cb.ax.tick_params(labelsize=6.5, width=0.6)
cb.outline.set_linewidth(0.6)
letter(1, 190, 'g')

# h: cross-day identification per participant (RF)
ax = ax_mm(84, 196, 86, 24)
v = PP.loc['RF'].values; ps = np.arange(1, 11)
ax.bar(ps, v, 0.62, color='#08519C', hatch='////', edgecolor='white', lw=0)
ax.axhline(0.1, ls='--', c='k', lw=0.7, label='Chance (0.10)')
ax.axhline(v.mean(), ls='-', c='#E6550D', lw=0.8, label=f'Mean of participants ({v.mean():.2f})')
ax.set_xticks(ps); ax.set_xlabel('Participant'); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1.0])
ax.set_ylabel('Day 1 → Day 2\naccuracy (RF)')
ax.legend(loc='upper right', ncol=2, bbox_to_anchor=(1, 1.08), handlelength=1.5)
letter(70, 190, 'h')

for ext, kw in [('pdf', {}), ('eps', {}), ('tif', dict(dpi=600, pil_kwargs={'compression': 'tiff_lzw'})), ('png', dict(dpi=450))]:
    fig.savefig(os.path.join(OUT, f'Fig8.{ext}'), **kw)
p = os.path.join(OUT, 'Fig8.tif'); Image.open(p).convert('RGB').save(p, compression='tiff_lzw', dpi=(600, 600))
print('Fig 8 written')
