"""
Валидация: N эпизодов в симуляции, вывод accuracy (доля успешных).
Чекпоинт должен лежать в ckpt/<run_name>/ (скачай с Dexter или положи локально).

Пример:
  python scripts/validate_accuracy.py policy.ckpt_name=1771602945-cautious-adder env_runner.num_episodes=100
"""
import sys
import os
_diffusion_policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "diffusion_policy")
if os.path.exists(_diffusion_policy_path) and _diffusion_policy_path not in sys.path:
    sys.path.insert(0, _diffusion_policy_path)

# PyTorch 2.7+: _refresh_per_optimizer_state в torch.amp.grad_scaler, Composer импортирует из torch.cuda.amp
import torch.cuda.amp.grad_scaler as _cuda_gs
if not hasattr(_cuda_gs, "_refresh_per_optimizer_state"):
    try:
        from torch.amp.grad_scaler import _refresh_per_optimizer_state
        _cuda_gs._refresh_per_optimizer_state = _refresh_per_optimizer_state
    except ImportError:
        pass

import json
import inspect
from pathlib import Path

import hydra
import wandb
from omegaconf import OmegaConf, open_dict
from pfp import set_seeds, REPO_DIRS
from pfp.envs.rlbench_runner import RLBenchRunner
from pfp.policy.base_policy import BasePolicy


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: OmegaConf):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    set_seeds(cfg.seed)
    wandb.init(mode="disabled")

    ckpt_path = REPO_DIRS.CKPT / cfg.policy.ckpt_name
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Download it from Dexter (e.g. scp ... ckpt/<run_name> ./ckpt/) or set policy.ckpt_name.")
        return

    with open_dict(cfg):
        train_cfg = OmegaConf.load(ckpt_path / "config.yaml")
        cfg.model = train_cfg.model
        cfg.env_runner.env_config.task_name = train_cfg.task_name
        cfg.env_runner.env_config.obs_mode = train_cfg.obs_mode
        cfg.env_runner.env_config.use_pc_color = train_cfg.dataset.use_pc_color
        cfg.env_runner.env_config.n_points = train_cfg.dataset.n_points
        cfg.env_runner.env_config.vis = False
        cfg.env_runner.env_config.headless = True

    policy_class = hydra.utils.get_class(train_cfg.model._target_)
    load_kwargs = {
        "ckpt_name": cfg.policy.ckpt_name,
        "ckpt_episode": cfg.policy.get("ckpt_episode", "latest"),
        "num_k_infer": cfg.policy.get("num_k_infer", 50),
        "flow_schedule": cfg.policy.get("flow_schedule", None),
        "exp_scale": cfg.policy.get("exp_scale", None),
        "subs_factor": cfg.policy.get("subs_factor", 1),
        "phase_conditioning": getattr(cfg, "phase_conditioning", None),
        "phase_prediction": getattr(cfg, "phase_prediction", None),
        "phase_rollout": getattr(cfg, "phase_rollout", None),
    }
    sig = inspect.signature(policy_class.load_from_checkpoint)
    if "meanflow_multistep_infer" in sig.parameters:
        load_kwargs["meanflow_multistep_infer"] = bool(cfg.policy.get("meanflow_multistep_infer", False))
    policy: BasePolicy = policy_class.load_from_checkpoint(**load_kwargs)
    # Runtime toggle for MeanFlow if the loader signature does not expose this kwarg.
    if bool(cfg.policy.get("meanflow_multistep_infer", False)) and hasattr(policy, "set_meanflow_multistep_infer"):
        policy.set_meanflow_multistep_infer(True)
        policy.set_num_k_infer(int(cfg.policy.get("num_k_infer", 1)))
    mm_schedule = cfg.policy.get("momentum_meanflow_schedule", None)
    if mm_schedule is not None and hasattr(policy, "set_momentum_meanflow_schedule"):
        policy.set_momentum_meanflow_schedule(str(mm_schedule))
    print(f"[model] class={policy.__class__.__name__} target={train_cfg.model._target_}")
    print(f"[model] num_k_infer={getattr(policy, 'num_k_infer', None)}")
    if hasattr(policy, "mm_cfg"):
        print(
            f"[model] momentum_meanflow schedule={policy.mm_cfg.momentum_meanflow_schedule} "
            f"lambda_correct={policy.mm_cfg.lambda_correct} dt_min={policy.mm_cfg.dt_min} "
            f"sampler_mode={getattr(policy, 'sampler_mode', None)}"
        )
    if hasattr(policy, "meanflow_enabled"):
        print(
            f"[model] meanflow.enabled={getattr(policy, 'meanflow_enabled', None)} "
            f"one_step={getattr(policy, 'meanflow_one_step', None)} "
            f"multistep_infer={getattr(policy, 'meanflow_multistep_infer', None)} "
            f"sampler_mode={getattr(policy, 'sampler_mode', None)} "
            f"meanflow_nfe={getattr(policy, 'meanflow_nfe', None)}"
        )
    env_runner = RLBenchRunner(**cfg.env_runner)
    success_list, steps_list, steps_per_episode, diagnostics = env_runner.run(
        policy, return_diagnostics=True
    )

    n = len(success_list)
    n_success = sum(success_list)
    acc = n_success / n if n else 0.0
    print(f"Accuracy: {n_success}/{n} ({100.0 * acc:.1f}%)")
    if steps_list:
        avg_steps = sum(steps_list) / len(steps_list)
        print(f"Avg steps (successful): {avg_steps:.1f}")

    export = {
        "ckpt_name": str(cfg.policy.ckpt_name),
        "ckpt_episode": str(cfg.policy.get("ckpt_episode", "latest")),
        "task_name": str(cfg.env_runner.env_config.task_name),
        "seed": int(cfg.seed),
        "num_episodes": int(n),
        "max_episode_length": int(cfg.env_runner.max_episode_length),
        "num_success": int(n_success),
        "accuracy": float(acc),
        "avg_steps_successful": float(sum(steps_list) / len(steps_list)) if steps_list else None,
        "mean_inference_ms": float(diagnostics.get("mean_inference_ms", 0.0)),
        "std_inference_ms": float(diagnostics.get("std_inference_ms", 0.0)),
        "nfe_per_action": float(diagnostics.get("nfe_per_action", 0.0)),
        "mean_episode_time_s": float(diagnostics.get("mean_episode_time_s", 0.0)),
        "policy_inference_diagnostics": diagnostics.get("policy_inference_diagnostics", {}),
        "sampler_mode": str(getattr(policy, "sampler_mode", policy.__class__.__name__)),
        "phase_conditioning": str(getattr(cfg, "phase_conditioning", None)),
        "phase_prediction": str(getattr(cfg, "phase_prediction", None)),
        "episodes": [
            {"episode_idx": i, "success": bool(s), "steps": int(steps_per_episode[i])}
            for i, s in enumerate(success_list)
        ],
    }
    out_json = getattr(cfg, "results_json", None)
    if out_json:
        out_path = Path(out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
        print(f"Wrote results JSON: {out_path}")

    return success_list


if __name__ == "__main__":
    main()
