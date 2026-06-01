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
