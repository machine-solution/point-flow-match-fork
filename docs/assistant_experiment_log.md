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
