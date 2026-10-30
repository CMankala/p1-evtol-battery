# P1 — Uncertainty-aware hybrid physics-informed battery prognostics for eVTOL mission energy reserves

This is the starter repository. Weeks 1–4 of the P1 plan reproduce the PHM 2021 hybrid physics-informed baseline (Nascimento, Corbetta, Kulkarni & Viana). P1's extensions are built on top of it after that.

## Layout

```
src/p1/models/hpinn.py   NASA Li-ion electrochemistry model (Daigle & Kulkarni 2013, as in ProgPy
                         BatteryElectroChemEOD v1.9) in PyTorch, with the hybrid option:
                         MLP (8-4-1 tanh) replaces the + electrode Redlich-Kister term, a linear
                         neuron replaces the - electrode term; q_Mobile and R_o are trainable.
src/p1/data/pcoe_rw.py   loader: NASA PCoE randomized battery usage (.mat, random-walk loading)
src/p1/data/cmu_evtol.py loader: CMU eVTOL battery data (CSV per cell, mission cycles)
scripts/train_hpinn.py   training: synthetic self-test | pcoe_rw | cmu_evtol
configs/baseline.yaml    run configuration
slurm/sweep_hpinn.sbatch Augie array job: 3 widths x 2 learning rates x 5 seeds
tests/test_hpinn.py      physics sanity, hybrid pre-training, and a check against ProgPy
```

## Setup (Augie or laptop)

```bash
python3 -m venv ~/venvs/p1 && source ~/venvs/p1/bin/activate
pip install -r requirements.txt          # on Augie, load the cluster's CUDA/python modules first
pytest -q                                # 4 tests; test_vs_progpy compares against NASA's ProgPy
python scripts/train_hpinn.py --source synthetic --epochs 150 --out runs/smoke
```

The synthetic run is the self-test. It generates curves from an "aged" cell (q_Mobile = 6800, R_o = 0.15 Ω; the defaults are 7600 and 0.117 Ω) and checks that the hybrid model recovers those values. If it cannot, do not move on to real data.

**Status (26 Oct 2026, CPU, 8 random-walk profiles of 2,400 s, 5 mV noise, 300 epochs ≈ 2.5 min):**

- `pytest`: 5/5 pass. The pure-physics path in `euler` mode matches NASA ProgPy `BatteryElectroChemEOD` to 2e-15 V over 3,000 s. The default `exp` integrator differs only during the first ~5 s of the ohmic transient (max 7.6 mV, mean 0.05 mV).
- **Self-test, two-stage training with the R_o prior:**

  | Run | q_Mobile (truth 6,800) | R_o (truth 0.150 Ω) | held-out RMSE |
  | --- | --- | --- | --- |
  | before the fix (joint training only) | 6,817 (+0.25%) | 0.139 (−7.4%) | 18.7 mV |
  | two-stage + prior, seed 0 | 6,802 (+0.02%) | 0.148 (−1.1%) | 8.0 mV |
  | two-stage + prior, seed 1 | 6,801 (+0.02%) | 0.147 (−1.7%) | 9.4 mV |

- **Identifiability.** The positive-electrode network can absorb part of the ohmic drop, so R_o and the network are not fully identifiable from discharge voltage alone. The training script now fixes this:
    1. **Stage 1 (60% of epochs):** fit the aging parameters only, with the networks frozen at their Redlich–Kister pre-training.
    2. **Stage 2:** train jointly, with a log-normal prior on R_o around its stage-1 value (σ = 0.1 in log space, λ = 1e-3, after Corbetta et al. Eq. 3), an L2 anchor on network weights, and R_o bounds of (0.005, 1.0) Ω.
    3. Keep the best iterate.

  Network inputs are centred (2x − 1) and pre-training runs 6,000 cosine-annealed steps. This cut the pre-training fit error from 16 mV to 5 mV RMS.

## Real data: how training works (updated 27 Oct 2026)

- **One aging state per sequence.** Each sequence gets its own aging state (q_Mobile, R_o), and the Redlich–Kister networks are shared across sequences. A NASA RW cell loses about half its capacity over its life (RW3: 2.01 → 1.06 Ah over 44 reference discharges), so one set of aging parameters cannot fit every sequence.
- **No truncation.** Sequences keep their full length: shorter ones are zero-current padded and the padding is masked out of the loss, so every end-of-discharge region is used.
- **Outputs:**
    - `result.json`: overall RMSE, max error, per-sequence RMSE, and the correlation between fitted q_Mobile and measured capacity, which should be close to +1;
    - `aging.csv`: the fitted aging trajectory, the raw material for the PHM 2021-style aging maps.
- **CMU missions** are found as contiguous discharge runs of 600–3,000 s with a takeoff current of at least 8 A in the first 120 s. `cycleNumber` is not a reliable delimiter in VAH files.
- **Mock-cell check:** 6 reference discharges with known capacity fade from 7,600 to 5,000 C. Fitted R_o tracked the truth throughout; q_Mobile tracked well for the first four sequences and was overestimated for the two most aged, because a 3.2 V cutoff truncates their depletion. Overall RMSE was 13.9 mV, and the correlation between fitted q_Mobile and measured capacity was 0.98.

## New cell types: identify constants first (CMU eVTOL)

The ProgPy constants describe a ~2 Ah, low-rate 18650. On the CMU VTC6 cells (3.0 Ah, 4–21 A, up to 68 °C), hybrid training with those constants failed: RMSE 229 mV, q_Mobile driven to about 4.5 Ah. Identify the cell first:

```
python scripts/fit_cell_params.py --path data/VAH01.csv --n-fit 6 --n-holdout 6 --out runs/vah01_params
python scripts/train_hpinn.py --source cmu_evtol --path data/VAH01.csv --cell-params runs/vah01_params/cell_params.json --dt 5 --epochs 300 --out runs/vah01_hpinn
```

`fit_cell_params.py` fits the pure-physics model to the first missions of the fresh cell. It frees q_Mobile, R_o, S_n, S_p, k_n, k_p, t_Diffusion, t_o, t_sn, t_sp and U0p, and reports RMSE on held-out missions.

**Synthetic check:** a cell with known non-default constants, 16 A takeoff and landing, 3 mV noise. RMSE fell from 813 mV with the default constants to 8.5 mV on the fit missions and 8.3 mV on held-out missions. Hybrid training from the identified constants then started at 26.6 mV instead of 667 mV.

**Caveat:** the voltage fit is good, but some constants are not unique. S_n/k_n and S_p/k_p trade off, and q_Mobile came out 10% high because missions use only part of the capacity. Report the identified set as an equivalent parameterization, not as measured physical values. The CMU capacity-check (RPT) cycles could pin q_Mobile and are a later addition.

## Out-of-sample evaluation (`scripts/evaluate_hpinn.py`)

- **`random-walk`:** blind prediction of the cell's random-walk discharges (0.5–4 A random loads), which training never sees. Nothing is refitted: each discharge's aging state is interpolated from the bracketing reference discharges. Reports voltage RMSE, max error and end-of-discharge (EOD) time error. To measure late EOD predictions, the load is extended past the end of the data instead of being clipped. This test is comparable with Corbetta et al.'s blind random-load results.
- **`cross-cell`:** the networks from cell A are frozen and only the aging states are fitted on cell B. The script compares this with a physics-only (Redlich–Kister) model given the same aging-only fit.

**Mock-cell check:**

- **Random-walk:** RMSE 18.4 ± 8.6 mV. EOD error was 0–20 s on lightly aged discharges but +450 to +930 s late on the most aged ones. These are the sequences whose fitted capacity is overestimated because of the 3.2 V cutoff. In P1 this is the reserve-energy safety problem: late EOD means an over-optimistic reserve.
- **Cross-cell:** RMSE 16.6 mV with the transferred networks against 19.8 mV for physics-only.

**Fix (27 Oct, evening).** In the real files, each random-walk discharge ends with a one-hour "rest post random walk discharge" step, and the first version wrongly included it in the discharge. That gave EOD errors of about −3,600 s and about +(block length): the latter because the extension current became NaN when the last 10 minutes were rest. Discharges now:
- end at that marker, without the rest;
- require the preceding charge to have reached ≥ 4.1 V and the last pulse to end ≤ 3.3 V (rejections are counted);
- take measured EOD as the end of the last pulse and extend the load at that pulse's current.

On a mock with this structure, all 21 blocks are usable, the median EOD error is 70 s and the mean absolute error is 272 s.

**Fix (27 Oct, late).** Early in the life of RW3–RW5 the tester went straight from the random-walk discharge into the next charge, with no "rest post" marker, so those discharges were counted as incomplete and early life was not tested. A discharge ended by a charge is now kept when it passes the same checks (charged to ≥ 4.1 V before, last pulse ≤ 3.3 V). The log line reports how many ended each way. Mock with markers removed from the first half of life: 21 usable (13 marker, 8 charge-ended), median EOD error 75 s.

## Week-by-week (P1 weeks 1–4)

1. **Get the data:**
    - NASA PCoE randomized battery usage data (RW files): load with `python -m p1.data.pcoe_rw RW9.mat`.
    - CMU eVTOL data (Bills et al., *Scientific Data* 2023): run `python -m p1.data.cmu_evtol VAH01.csv` and fix `COLUMN_MAP` if any column is unmatched.
2. **Reproduce:** train on reference discharges and report voltage RMSE and MAD. The targets come from Corbetta et al. Table 3: RMSE 0.0253–0.0934 V and MAD 0.1435–0.2939 V across 8 batteries. The PHM 2021 paper reports MSE, not RMSE, so compare like with like (see `p1_reading_notes.md`, cross-paper synthesis).
3. **Fill the gaps the papers leave open:** optimizer, epochs and batch size are not reported. Record your choices in `configs/` and say so in the methods section.
4. **Sweep on Augie:** `sbatch slurm/sweep_hpinn.sbatch`.

## Next steps (P1 contributions, weeks 5–10)

- **Uncertainty:** deep ensembles (5–10 seeds from the sweep) and MC dropout in `vint_p`. Calibrate with PICP and CRPS.
- **Aging:** fleet-level maps q_max(E) and R_o(E) on cumulative energy, as in PHM 2021 §3.
- **Mission profiles:** a power-required model for the UAM multirotor (Pradeep et al. Eq. 17, Table 1) gives pack power, which `PackModel` (to write) converts to cell current.
- **Output metric:** P(landing reserve < regulatory minimum) under wind, departure-time and aging uncertainty.
- **Baselines:** ProgPy electrochemical model with UKF; 1-RC ECM with EKF; LSTM; energy-based model (Pradeep); the NDARC-style normalized model (Horton et al. 2024).

## Caveats

- The physics constants are ProgPy's 18650 defaults. Real cells need fitted parameters; PHM 2021 learns only q_max and R_o on top of these.
- Integration: the default `exp` integrator is stable at dt = 10 s. Plain Euler (`integrator="euler"`, identical to ProgPy) oscillates once dt exceeds t_o ≈ 6.1 s, so use it only with dt ≤ 1–2 s. Check that results do not change between dt = 10 s and dt = 1 s.
- End-of-discharge threshold: ProgPy uses 3.0 V, while the PCoE random-walk discharges stop at 3.2 V. Set `data.v_eod` per dataset.
- Data-loader field names are the expected ones. Both loaders print what they found and stop on a mismatch rather than guessing.

## Whole-life random-walk results and the two follow-up tools (27 Oct, night)

Real-data rerun with early life included: usable discharges RW3 744, RW4 639, RW5 666, RW6 403.
Median EOD error 36 / 508 / 57 / 17 s. Never more than 72 s late; 22 of 80 more than 250 s early.

- **Diagnostics** (`evaluate_hpinn.py random-walk`, always on): per discharge, the current and the
  measured voltage at the moment the model crossed the cut-off; overall, voltage error vs current
  (`err_vs_current`: slope in V/A and mean bias per current band). A clearly negative slope means
  too much sag under load (R_o or overpotential too high). A measured voltage just above 3.2 V at
  the predicted EOD means the miss comes from a near-threshold plateau, not a bad fit.
- **`--ro-scale`** (diagnostic only, never for reported numbers): scales the interpolated R_o at
  test time. Results go to `eval_random_walk_ro<scale>`.
- **`train_hpinn.py --add-rw N`**: also trains on N random-walk discharges (0.5–4 A). Their aging
  state is interpolated from the reference states (the same rule used at test time), so they teach
  the networks high-current behaviour without adding free parameters. They are recorded in
  `result.json` (`train_rw_steps`) and excluded from the blind test. `rmse_v` stays the
  reference-discharge RMSE, so it stays comparable with runs without `--add-rw`.
  Mock check (6 added): blind RMSE 32.9 → 27.4 mV, median EOD error 75 → 15 s.

## Mixed-current training result and the ensemble (28 Oct)

Blind random-walk, seed 0, reference-only → +12 random-walk discharges in training:
RMSE 49/54/52/33 → 43/44/44/31 mV; median EOD error 36/508/57/17 → 20/56/42/24 s;
discharges more than 250 s early 22 → 16 of 80. Remaining bias is a current-independent
offset (−13 to −31 mV at low/medium current), not resistance. **`--add-rw 12` is now the
standard setting.**

- `evaluate_hpinn.py ensemble --runs <seed folders> --path data/RWx.mat`: ensemble-mean
  accuracy; EOD interval = range of the members, with coverage (nominal (n−1)/(n+1), 67% for 5
  members) and median width; voltage-band coverage; and the earliest member as a conservative
  EOD (`conservative_eod_late_frac` = how often even that is late).
- `random-walk` now also reports the bias in the first and last quarter of each discharge
  (`bias_by_phase_mv`). If the bias grows towards the end, the capacity is low; if it is there from
  the start, the open-circuit curve is offset.

## Estimate-then-predict (`--assim-s`, 28 Oct)

`random-walk` and `ensemble` take `--assim-s 600`: refine (q_Mobile, R_o) on the first 600 s of each
discharge (MAP fit with priors centred on the interpolated state; networks frozen), then forecast
the rest. This addresses the per-discharge offset (whole curve 50–75 mV low on the big misses),
which comes from the aging interpolation and which seed ensembles cannot see. Results go to
`eval_random_walk_assim600` / `runs/ensemble_RWx_assim600`; `rmse_v_forecast` is the error after the
window only. Mock: median EOD error 75 → 15 s, forecast RMSE 33 → 12 mV.

## Conformal calibration (`ensemble --calib 40`, 28 Oct)

RW3 seed ensemble: EOD coverage 35% vs nominal 67%, median width 65 s; the seeds barely differ.
`--calib N` takes N further discharges (disjoint from training and test, spread over life),
computes residuals of the ensemble-mean EOD, and reports a split-conformal two-sided interval
and a one-sided **planning EOD** (`plan_eod_s`, early with probability ≥ 1−alpha). Summary:
`conformal.coverage`, `width_s_median`, `plan_late_frac`, `plan_margin_s_median`. Default alpha 0.1.

## Final ensemble results and prediction-time evaluation (28 Oct)

Conformal planning EOD (alpha 0.1, 40 calibration discharges), blind ensemble: margin 51–132 s,
late 0/0/4/0 of 20 on RW3–RW6 (4/80 pooled), reserve cost 63–172 s. Refinement on the first 600 s
made accuracy worse (the first 10 min carry little capacity information) — kept as an ablation.

`ensemble --tp-frac 0.5`: refine each member's aging state on the data up to t_P = 0.5·t_EOD, forecast
EOD after t_P; reports `rul_rel_err_median` and `alpha_lambda_acc_20pct` alongside the conformal summary.

## Baselines, mission layer, out-of-distribution, tables (28 Oct, evening)

- `train_hpinn.py --model {hybrid,physics,ecm,lstm}`. `physics`: Redlich–Kister model, aging only.
  `ecm`: 1-RC equivalent circuit, OCV polynomial initialised from the electrochemistry model.
  `lstm`: data-driven, inputs current / coulomb-counted SOC / capacity. All four go through the
  same `evaluate_hpinn.py ensemble --calib` (conformal planning EOD), so the tables compare like
  with like. Mock check: all train and evaluate; the LSTM extrapolates poorly past the data.
- `train_hpinn.py --ambient-c 40` for the 40 °C cells (sets tb0; saved and reused by evaluation).
- `scripts/mission_reserve.py`: CMU missions — blind voltage, conformal bound on the landing
  voltage, predicted and planning reserve at loiter current, reserve risk over life.
- `scripts/make_tables.py`: every result.json → CSV tables for the manuscript.
- `evaluate_hpinn.py` now prints calibration progress (`calib j/N`).

## Baseline result and the warm-start fix (28 Oct, night)

Blind random-walk with conformal planning EOD: the physics-only model and the 1-RC ECM had LOWER
voltage RMSE than the hybrid (35–39 / 27–36 vs 30–43 mV). Cause: the hybrid's stage 1 (60% of 300
epochs) stopped before the aging states converged, and stage 2 trains them at 0.1× lr. Fix:
`--init-aging-from runs/rwX_phys` copies the converged physics-only aging states and skips stage 1,
so the hybrid = physics solution + network correction. Mock: 22.5 → 15.3 mV (physics 18.5 mV).

## 29 Oct: warm-start hybrid is the final configuration; CMU reserve floors

Warm-start hybrid (5 seeds): in-sample 19.3/18.4/20.3/10.9 mV (physics-only 32.7/31.1/32.3/20.4).
Blind + conformal: median EOD 25/63/32/21 s, q_plan 56/114/56/71 s, late 4/80, reserve cost 93 s mean
(physics 116, ECM 124, LSTM 276). Blind voltage RMSE is still 1–6 mV worse than physics-only/ECM.
`mission_reserve.py --v-cut 3.0 2.8 2.7` evaluates several reserve floors from one set of simulations
(violations measured/planned/missed/false alarms per floor; default reserve thresholds 2/5/8 min).
