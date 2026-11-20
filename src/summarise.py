"""Summary tables from the result CSVs.
Usage: python -I summarise.py <results_dir>
"""
import sys, os, glob
import numpy as np, pandas as pd
from collections import Counter
R = sys.argv[1]
CHANCE = {'A_sameday_30class': 1 / 30, 'A_sameday_person': 1 / 10, 'B_person_crossday': 1 / 10, 'C_position_LOSO': 1 / 3}


def load(pattern):
    fs = [f for f in glob.glob(os.path.join(R, pattern)) if not f.endswith(('_preds.csv',))]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


main = pd.concat([load('classic_all.csv'), load('cnn_*.csv'), load('cnnaug_*.csv')], ignore_index=True)
order = ['KNN', 'SVM', 'RF', 'GB', 'MLP', 'CNN', 'CNN_aug']
main['subset'] = main.get('subset', 'all')
main['subset'] = main['subset'].fillna('all')
t = main.groupby(['subset', 'protocol', 'model']).agg(acc=('acc', 'mean'), sd=('acc', 'std'), f1=('f1', 'mean'), walk=('walk_acc', 'mean'), runs=('acc', 'size'))
t = t.reset_index(); t['chance'] = t.protocol.map(CHANCE)
t['model'] = pd.Categorical(t.model, order, ordered=True)
t = t.sort_values(['subset', 'protocol', 'model'])
t.round(3).to_csv(os.path.join(R, 'summary_main.csv'), index=False)
print(t.round(3).to_string(index=False))

# Protocol B: day-level fusion (all positions of one person-day) and per-person accuracy
preds = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, '*_preds.csv'))
                   if os.path.basename(f).startswith(('classic_all', 'cnn_', 'cnnaug_'))], ignore_index=True)
B = preds[(preds.protocol == 'B_person_crossday') & (preds.subset == 'all')].copy()
B['direction'] = B.fold.str.split('_').str[1]
rows = []
for (m, s, dirn), g in B.groupby(['model', 'seed', 'direction']):
    for p, gp in g.groupby('person'):
        rows.append(dict(model=m, seed=s, direction=dirn, person=p, n_pos=gp.position.nunique(),
                         step_acc=(gp.y_true == gp.y_pred).mean(),
                         day_correct=Counter(gp.y_pred).most_common(1)[0][0] == gp.y_true.iloc[0]))
D = pd.DataFrame(rows)
dl = D.groupby('model').agg(day_acc=('day_correct', 'mean'), n=('day_correct', 'size')).round(3)
print('\nB day-level (all positions of one test day combined, majority vote):'); print(dl.to_string())
pp = D.groupby(['model', 'person']).step_acc.mean().unstack().round(2)
print('\nB per-person step accuracy:'); print(pp.to_string())
dl.to_csv(os.path.join(R, 'summary_B_daylevel.csv')); pp.to_csv(os.path.join(R, 'summary_B_perperson.csv'))
