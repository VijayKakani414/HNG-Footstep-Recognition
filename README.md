# Machine-learning analysis of HNG footstep signals

Code, data and results for the machine-learning part of
*"Multifunctional bismuth tungstate/polydimethylsiloxane hybrid nanogenerator for energy harvesting, smart security,
and machine learning-enabled human motion recognition"* (Section 2.6, Fig. 8, Table 1, Supplementary Tables S1–S4 and Figs. S9–S12).

## Data

`data/raw/` holds the walking recordings: ten participants, each recorded on two separate days, with one 20 s walk per
plantar region (heel, meta, toe) and day. Each Excel file has two columns, time (s) and open-circuit voltage (V), sampled at
about 260 Hz. File names follow `Person<P>_Day<D>_<Region>.xlsx`. The 58 recordings and the number of footsteps used from each
are listed in `results/dataset_table.csv` (Table S1).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `src/data.py` | 40 Hz zero-phase low-pass filter, median removal, automatic footstep detection, non-overlapping 0.75 s footstep segments, 18 features per footstep |
| 2 | `src/protocols.py` | The three evaluation protocols (training and test data never overlap) |
| 3 | `src/eval_classic.py` | KNN, SVM, RF, GB, MLP (fixed hyperparameters) under all protocols |
| 4 | `src/eval_cnn.py` | 1-D CNN with explicit train / validation / test split (validation only for early stopping); optional augmentation |
| 5 | `src/eval_nested.py` | Additional attempts: nested cross-validated tuning, per-participant-day calibration |
| 6 | `src/permtest.py` | Recording-level label-permutation tests |
| 7 | `src/control_gap.py` | Recording-fingerprint control: quiet segments between footsteps |
| 8 | `src/summarise.py` | Summary tables |
| 9 | `figures/fig8.py`, `figures/si_figs.py` | Fig. 8 and Figs. S9–S12 (PDF, EPS, 600 dpi TIFF) |

### Evaluation protocols

| Protocol | Task | Test data | Training data | CNN validation data |
|---|---|---|---|---|
| Same day | 30 classes (participant × region) or 10 participants | one of five contiguous footstep groups of every recording of that day | the other groups | one further group |
| Cross-day | 10 participants, one model per region | all recordings of the other day | all recordings of one day (both directions) | last 20 % of every training recording |
| Cross-participant | 3 plantar regions | all recordings of the held-out participant | the other participants | all recordings of one further participant |

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh            # about 2 hours on a 2-core CPU; results/ and output/ are rewritten
```

Tested with Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, scikit-learn 1.9, TensorFlow 2.21 and Matplotlib 3.11.
Seeds are fixed; differences of about ±0.01 in accuracy can occur with other library versions or file orderings.

## Main results (accuracy per footstep, mean ± SD)

| Protocol | Best model | Accuracy | Chance |
|---|---|---|---|
| Same day, 30 classes | RF | 0.81 ± 0.08 | 0.03 |
| Same day, 10 participants | RF | 0.81 ± 0.06 | 0.10 |
| Cross-day participant identification | RF | 0.38 ± 0.05 (0.52 per participant-day) | 0.10 |
| Cross-participant plantar region | CNN | 0.41 ± 0.15 (not significant) | 0.33 |

Full results: `results/summary_main.csv`; robustness, permutation tests and additional attempts: `results/classic_noP1P2.csv`,
`results/classic_noclip.csv`, `results/perm_*.csv`, `results/nested_*.csv`, `results/control_gap.csv`.

## Layout

```
data/raw/        58 recordings (.xlsx)
src/             analysis code
figures/         figure scripts and the panel-a schematic
results/         result tables (.csv)
output/          figures (PDF, EPS, TIFF)
run_all.sh       runs everything
```
