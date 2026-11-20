#!/bin/bash
# Reproduces every number and figure of the machine-learning analysis.
# Usage: bash run_all.sh          (about 2 hours on a 2-core CPU)
set -euo pipefail
cd "$(dirname "$0")"
D=results/dataset.pkl
python -I src/data.py data/raw $D > results/dataset_log.txt
python -c "import pickle; pickle.load(open('$D','rb'))['table'].to_csv('results/dataset_table.csv', index=False)"
# classical models: all protocols, robustness subsets
python -I src/eval_classic.py $D results/classic_all.csv all
python -I src/eval_classic.py $D results/classic_noP1P2.csv noP1P2 B_person_crossday
python -I src/eval_classic.py $D results/classic_noclip.csv noclip
# CNN (train / validation / test) and CNN with augmentation
for P in A_sameday_30class A_sameday_person B_person_crossday C_position_LOSO; do
  python -I src/eval_cnn.py $D results/cnn_$P.csv $P
done
SUBSET=noP1P2 python -I src/eval_cnn.py $D results/cnn_B_noP1P2.csv B_person_crossday
for P in B_person_crossday C_position_LOSO; do python -I src/eval_cnn.py $D results/cnnaug_$P.csv $P aug; done
SUBSET=noP1P2 python -I src/eval_cnn.py $D results/cnnaug_B_noP1P2.csv B_person_crossday aug
# additional attempts, permutation tests, recording-fingerprint control
for W in I1_A I1_B I1_C I4_C; do python -I src/eval_nested.py $D results/nested_$W.csv $W; done
python -I src/permtest.py $D A_sameday_30class RF 100 results/perm_A30_RF.csv
python -I src/permtest.py $D B_person_crossday RF 100 results/perm_B_RF.csv
SUBSET=noP1P2 python -I src/permtest.py $D B_person_crossday RF 100 results/perm_B_RF_noP1P2.csv
python -I src/permtest.py $D C_position_LOSO SVM 100 results/perm_C_SVM.csv
python -I src/control_gap.py $D results/control_gap.csv
# summaries and figures
python -I src/summarise.py results > results/summary_print.txt
python -I figures/fig8.py . figures/panel_a.png output
python -I figures/si_figs.py . output
echo "Done. Tables in results/, figures in output/."
