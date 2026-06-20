# Assistant Experiment Log

Этот файл веду я (assistant). Пользовательские файлы не менять.

## 2026-05-26 - Baseline K-sweep (key result)

Source run tag: `k_sweep_baseline_ep1500_seed5678_n100_k1_k10`

- `K=1`: success_rate `0.10` (`10/100`), mean_inference_ms `718.29`
- `K=10`: success_rate `0.28` (`28/100`), mean_inference_ms `3179.22`
- Speedup (`K=1` vs `K=10`): `~4.43x` faster by mean inference ms
- Quality delta: `K=1` is `-18 pp` vs `K=10` (strong degradation)
- Decision recorded: Case A -> baseline one-step is not enough; move to MeanFlow branch

## 2026-05-26 - Milestone study (baseline vs phased, n=50, max_steps=120)

Source run tags:
- `1779122560-baseline-many-ckpts_seed5678_n50_max120`
- `1779258120-phased-many-ckpts_seed5678_n50_max120`

- ep300: baseline `7/50` (`14%`), phased `4/50` (`8%`)
- ep600: baseline `15/50` (`30%`), phased `8/50` (`16%`)
- ep900: baseline `13/50` (`26%`), phased `10/50` (`20%`)
- ep1200: baseline `13/50` (`26%`), phased `17/50` (`34%`)
- ep1500: baseline `15/50` (`30%`), phased `13/50` (`26%`)

## Log format (for next entries)

For each run, record:
- command/run-id
- fixed settings: seed, num_episodes, max_episode_length
- metrics: success_count, success_rate, mean/std inference latency, nfe_per_action
- short conclusion (1 line)

## 2026-05-29 - MeanFlow checkpoint `meanflow_open_fridge_1365` (validation + timing)

Reference baseline for comparison:
- baseline `K=1`: success `10/100` (`10%`), mean_inference_ms `718.29`
- baseline `K=10`: success `28/100` (`28%`), mean_inference_ms `3179.22`

Measured (timing):
- command: `python scripts/benchmark_inference_latency.py --ckpt-name meanflow_open_fridge_1365 --ckpt-episode latest --num-k-infer 1`
- output csv: `results/efficiency/latency_meanflow_open_fridge_1365.csv`
- mean latency `25.51 ms`, std `1.24 ms`, p50 `25.26 ms`, p90 `26.67 ms`, `NFE/action=1.00`

Measured (accuracy):
- running command: `CONDA_ENV=pfp_env bash bash/run_validate_accuracy.sh meanflow_open_fridge_1365 100`
- output file: `results/validate_accuracy_meanflow_open_fridge_1365_20260529_204547.txt`
- result: success `6/100` (`6.0%`), avg steps (successful) `88.8`

Comparison vs baseline:
- quality vs baseline `K=1`: `6%` vs `10%` (`-4 pp`)
- quality vs baseline `K=10`: `6%` vs `28%` (`-22 pp`)
- speed vs baseline `K=1`: `25.51 ms` vs `718.29 ms` (`~28.2x` faster)
- speed vs baseline `K=10`: `25.51 ms` vs `3179.22 ms` (`~124.6x` faster)

## 2026-05-30 - MeanFlow checkpoint `meanflow_open_fridge_1365` (`max_episode_length=200`)

Measured (accuracy + runtime in rollout):
- command: `conda run -n pfp_env python scripts/validate_accuracy.py policy.ckpt_name=meanflow_open_fridge_1365 env_runner.num_episodes=100 env_runner.max_episode_length=200 ...`
- output json: `results/validate_accuracy_meanflow_open_fridge_1365_n100_max200.json`
- result: success `8/100` (`8.0%`), avg steps (successful) `74.75`
- rollout inference stats: `mean_inference_ms=29.57`, `std_inference_ms=11.91`, `nfe_per_action=1.0`
- rollout wall-clock: `mean_episode_time_s=45.97`

Comparison:
- vs same checkpoint at `max=120`: `8%` vs `6%` (`+2 pp`)
- vs baseline `K=1` (`10%`): `-2 pp`
- vs baseline `K=10` (`28%`): `-20 pp`

## 2026-06-07 - Shortcut checkpoint `shortcut_open_fridge_1385` (validation)

Measured (accuracy):
- command: `CONDA_ENV=pfp_env bash bash/run_validate_accuracy.sh shortcut_open_fridge_1385 100`
- output file: `results/validate_accuracy_shortcut_open_fridge_1385_20260607_184128.txt`
- result: success `30/100` (`30.0%`), avg steps (successful) `72.2`
- rollout time for full 100-episode run: `6:04:29`

Quick comparison:
- vs baseline `K=1` (`10%`): `+20 pp`
- vs baseline `K=10` (`28%`): `+2 pp`
- vs MeanFlow `meanflow_open_fridge_1365` at `max=120` (`6%`): `+24 pp`

## 2026-06-08 - MeanFlow `K`-sweep without retraining (`K=1,2,4,6,8,10`)

Measured:
- command: `conda run -n pfp_env python scripts/sweep_num_k_infer.py --ckpt-name meanflow_open_fridge_1365 --ckpt-episode latest --num-episodes 100 --max-episode-length 120 --ks 1,2,4,6,8,10 ...`
- output csv: `results/efficiency/k_sweep_meanflow_open_fridge_1365_n100_k1_2_4_6_8_10.csv`
- output json: `results/efficiency/k_sweep_meanflow_open_fridge_1365_n100_k1_2_4_6_8_10.json`

Results:
- `K=1`: success `10/100` (`10%`), mean_inference_ms `34.13`, nfe/action `1.0`
- `K=2`: success `10/100` (`10%`), mean_inference_ms `31.89`, nfe/action `1.0`
- `K=4`: success `6/100` (`6%`), mean_inference_ms `26.65`, nfe/action `1.0`
- `K=6`: success `6/100` (`6%`), mean_inference_ms `26.72`, nfe/action `1.0`
- `K=8`: success `11/100` (`11%`), mean_inference_ms `26.22`, nfe/action `1.0`
- `K=10`: success `9/100` (`9%`), mean_inference_ms `26.32`, nfe/action `1.0`

Conclusion:
- changing `policy.num_k_infer` for this MeanFlow checkpoint does not increase effective NFE (stays `1.0`), and quality remains in the `6-11%` band.

## 2026-06-08 - Per-action latency benchmark (algorithm runtime)

Measured:
- command: `conda run -n pfp_env python scripts/benchmark_inference_latency.py ...`
- output csv: `results/efficiency/latency_algorithms_20260608.csv`
- setup: `batch_size=1`, `warmup=10`, `timed_iters=50`

Per-step inference time (`infer_y`, ms/action):
- baseline FM (`1779122560-baseline-many-ckpts`, `K=10`): mean `194.30 ms`, p50 `193.11`, p90 `199.69`, `NFE/action=10`
- MeanFlow (`meanflow_open_fridge_1365`, `K=1`): mean `42.76 ms`, p50 `43.31`, p90 `44.60`, `NFE/action=1`
- Shortcut (`shortcut_open_fridge_1385`, `K=1`): mean `44.41 ms`, p50 `44.59`, p90 `46.19`, `NFE/action=1`

Quick speed comparison:
- MeanFlow vs baseline `K=10`: `~4.54x` faster per action
- Shortcut vs baseline `K=10`: `~4.37x` faster per action
- Shortcut vs MeanFlow: `~3.9%` slower per action (close)

## 2026-06-08 - MeanFlow latency recheck (variance investigation)

Reason:
- user noticed inconsistent MeanFlow latency values across runs.

Recheck command:
- `conda run -n pfp_env python scripts/benchmark_inference_latency.py --ckpt-name meanflow_open_fridge_1365 --ckpt-episode latest --num-k-infer 1` (3 repeats, defaults `warmup=20`, `timed_iters=100`)
- output csv: `results/efficiency/latency_meanflow_recheck_20260608.csv`

Recheck results:
- run1: mean `53.52 ms`
- run2: mean `52.69 ms`
- run3: mean `53.50 ms`
- stable band now: `~52.7-53.5 ms`

Environment check:
- `torch.cuda.is_available=False`, `device_count=0` in `pfp_env`.
- Important: latency numbers from different sessions are not directly comparable unless hardware/runtime (CPU vs GPU), warmup/timed-iters, and machine load are fixed.

## 2026-06-09 - Clarification: `Shortcut` K mismatch (accuracy vs latency)

Important:
- `Shortcut` accuracy run (`30/100`) was executed with `policy.num_k_infer=50` (from `conf/eval.yaml` defaults in `run_validate_accuracy.sh`).
- previous `Shortcut` latency entry (`~44 ms`) was measured separately at `K=1` (microbench), so it is not comparable to that `K=50` accuracy run.

Direct apples-to-apples latency check (same benchmark context):
- command: `scripts/benchmark_inference_latency.py` with `meanflow K=1` and `shortcut K=50`
- output csv: `results/efficiency/latency_k_compare_20260609.csv`
- MeanFlow `K=1`: mean `30.79 ms`, `NFE/action=1`
- Shortcut `K=50`: mean `1155.72 ms`, `NFE/action=50`
- ratio (`shortcut K=50` / `meanflow K=1`): `~37.5x` slower per action

## 2026-06-09 - Shortcut K-sweep (`K=1,2,5,10`) with accuracy + runtime

Measured:
- command: `conda run -n pfp_env python scripts/sweep_num_k_infer.py --ckpt-name shortcut_open_fridge_1385 --ckpt-episode latest --num-episodes 100 --max-episode-length 120 --ks 1,2,5,10 ...`
- output csv: `results/efficiency/k_sweep_shortcut_open_fridge_1385_n100_k1_2_5_10.csv`
- output json: `results/efficiency/k_sweep_shortcut_open_fridge_1385_n100_k1_2_5_10.json`

Results:
- `K=1`: success `12/100` (`12%`), mean_inference_ms `31.17`, nfe/action `1.0`
- `K=2`: success `13/100` (`13%`), mean_inference_ms `45.30`, nfe/action `2.0`
- `K=5`: success `19/100` (`19%`), mean_inference_ms `108.27`, nfe/action `5.0`
- `K=10`: success `24/100` (`24%`), mean_inference_ms `216.62`, nfe/action `10.0`

Conclusion:
- quality increases with `K` for this checkpoint (`12% -> 24%`), and per-action inference cost scales roughly linearly with `K`.

## 2026-06-11 - StepConditionedMeanFlow checkpoint `step_conditioned_meanflow_open_fridge_1392` (validation)

Measured (accuracy):
- command: `CONDA_ENV=pfp_env bash bash/run_validate_accuracy.sh step_conditioned_meanflow_open_fridge_1392 100`
- output file: `results/validate_accuracy_step_conditioned_meanflow_open_fridge_1392_20260611_073224.txt`
- runtime: `7:15:08` for `100` episodes
- result: success `13/100` (`13.0%`), avg steps (successful) `80.5`

Notes:
- eval path currently uses `policy.num_k_infer=50` from `conf/eval.yaml` unless overridden.

## 2026-06-11 - StepConditionedMeanFlow `K=1` vs `K=10` (explicit sweep)

Measured:
- command: `conda run -n pfp_env python scripts/sweep_num_k_infer.py --ckpt-name step_conditioned_meanflow_open_fridge_1392 --ckpt-episode latest --num-episodes 100 --max-episode-length 120 --ks 1,10 ...`
- output csv: `results/efficiency/k_sweep_step_conditioned_meanflow_open_fridge_1392_n100_k1_k10.csv`
- output json: `results/efficiency/k_sweep_step_conditioned_meanflow_open_fridge_1392_n100_k1_k10.json`

Results:
- `K=1`: success `14/100` (`14%`), mean_inference_ms `31.93`, nfe/action `1.0`
- `K=10`: success `18/100` (`18%`), mean_inference_ms `208.84`, nfe/action `10.0`

Conclusion:
- explicit `K` override works as expected (`nfe/action` matches `K`);
- higher `K` improves quality for this checkpoint (`+4 pp`), at ~`6.5x` higher per-action inference time.

## 2026-06-14 - MeanFlow true multistep K-sweep (`meanflow_multistep_infer=true`)

Context:
- fixes bug from `2026-06-08` sweep where `policy.num_k_infer` was ignored and `nfe/action` stayed `1.0` for all `K`;
- sampler mode: `meanflow_multistep` (`x <- x + dt * u`, `NFE/action = K`).

Measured:
- command: `conda run -n pfp_env python scripts/sweep_num_k_infer.py --ckpt-name meanflow_open_fridge_1365 --ckpt-episode latest --num-episodes 100 --max-episode-length 120 --seed 5678 --ks 1,2,4,6,8,10 --meanflow-multistep --resume-existing ...`
- output csv: `results/efficiency/k_sweep_meanflow_open_fridge_1365_n100_k1_2_4_6_8_10_multistep.csv`
- output json: `results/efficiency/k_sweep_meanflow_open_fridge_1365_n100_k1_2_4_6_8_10_multistep.json`
- log: `results/efficiency/k_sweep_meanflow_open_fridge_1365_n100_k1_2_4_6_8_10_multistep.log`
- run was interrupted once after `K=2`; resumed with `--resume-existing` and completed `K=4,6,8,10`.

Results:
- `K=1`: success `7/100` (`7%`), mean_inference_ms `30.04`, std `7.22`, nfe/action `1.0`, mean_episode_time_s `28.50`
- `K=2`: success `22/100` (`22%`), mean_inference_ms `45.32`, std `9.56`, nfe/action `2.0`, mean_episode_time_s `26.29`
- `K=4`: success `22/100` (`22%`), mean_inference_ms `80.07`, std `13.52`, nfe/action `4.0`, mean_episode_time_s `29.88`
- `K=6`: success `22/100` (`22%`), mean_inference_ms `117.59`, std `17.99`, nfe/action `6.0`, mean_episode_time_s `34.06`
- `K=8`: success `22/100` (`22%`), mean_inference_ms `156.62`, std `26.26`, nfe/action `8.0`, mean_episode_time_s `38.38`
- `K=10`: success `26/100` (`26%`), mean_inference_ms `194.01`, std `30.27`, nfe/action `10.0`, mean_episode_time_s `41.02`

Comparison vs old broken MeanFlow sweep (`2026-06-08`, one-step only):
- old `K=1`: `10%` at `34.13 ms`, nfe `1.0`; new multistep `K=1`: `7%` at `30.04 ms`, nfe `1.0` (one-step path unchanged in quality band);
- old sweep: all `K` had nfe `1.0` and SR `6-11%`; new multistep: nfe scales with `K`, SR jumps to `22%` at `K>=2`.

Comparison vs baseline / Shortcut (same `N=100`, `max=120`, seed `5678` where available):
- baseline `K=10`: `28%` at `~3179 ms` (old sweep csv) / `~194 ms` (microbench);
- Shortcut `K=10`: `24%` at `216.62 ms`, nfe `10.0`;
- MeanFlow multistep `K=10`: `26%` at `194.01 ms`, nfe `10.0` (similar quality to Shortcut `K=10`, similar per-action cost to baseline FM microbench `K=10`).

Conclusion:
- true multistep inference works (`nfe/action == K` verified in json diagnostics);
- large quality gain from `K=1` to `K=2` (`+15 pp`), then plateau `22%` until `K=10` (`26%`);
- per-action latency grows ~linearly (`~19-20 ms` per extra NFE);
- at `K=10`, MeanFlow multistep reaches near-Shortcut quality with comparable per-action inference time.
