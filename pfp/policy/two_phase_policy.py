from __future__ import annotations

import numpy as np

from pfp.policy.base_policy import BasePolicy


class TwoPhasePolicy(BasePolicy):
    """
    Wrapper-policy that switches from pre -> post based on *observed* gripper state.

    Switch criterion (вариант A):
      - look at current robot_state[9] (gripper openness)
      - if it is < gripper_thr for `closed_steps_to_switch` consecutive env steps
        -> switch to post policy
    """

    def __init__(
        self,
        pre: BasePolicy,
        post: BasePolicy,
        *,
        gripper_thr: float = 0.5,
        closed_steps_to_switch: int = 3,
    ) -> None:
        # BasePolicy ctor needs n_obs_steps/subs_factor, but we delegate all inference,
        # so we just mirror pre's settings.
        super().__init__(n_obs_steps=getattr(pre, "n_obs_steps", 2), subs_factor=getattr(pre, "subs_factor", 1))
        self.pre = pre
        self.post = post
        self.gripper_thr = float(gripper_thr)
        self.closed_steps_to_switch = int(closed_steps_to_switch)

        self._use_post = False
        self._closed_count = 0
        self._cur_step: int = -1
        self._switch_step: int | None = None

    def reset_obs(self):
        # Runner calls this once per episode
        self.pre.reset_obs()
        self.post.reset_obs()
        self._use_post = False
        self._closed_count = 0
        self._cur_step = -1
        self._switch_step = None

    def set_step(self, step_idx: int) -> None:
        """Optional hook: runners may set current env step index for logging."""
        self._cur_step = int(step_idx)

    def get_phase(self) -> str:
        return "post" if self._use_post else "pre"

    def get_switch_step(self) -> int | None:
        """Env step index at which we switched to post (None if never switched)."""
        return self._switch_step

    # We never call BasePolicy.infer_from_np on this wrapper. RLBenchRunner calls predict_action(),
    # so implement switching logic here.
    def predict_action(self, obs: np.ndarray, robot_state: np.ndarray) -> np.ndarray:
        g = float(np.asarray(robot_state).ravel()[9])
        is_closed = g < self.gripper_thr

        if not self._use_post:
            if is_closed:
                self._closed_count += 1
            else:
                self._closed_count = 0

            if self._closed_count >= self.closed_steps_to_switch:
                self._use_post = True
                if self._switch_step is None and self._cur_step >= 0:
                    self._switch_step = int(self._cur_step)
                # Start post policy fresh from current observation; BasePolicy will pad history.
                self.post.reset_obs()

        policy = self.post if self._use_post else self.pre
        return policy.predict_action(obs, robot_state)

    # BasePolicy abstract API (not used by this wrapper)
    def _norm_obs(self, obs):  # pragma: no cover
        raise NotImplementedError("TwoPhasePolicy is a wrapper; call predict_action()")

    def _norm_robot_state(self, robot_state):  # pragma: no cover
        raise NotImplementedError("TwoPhasePolicy is a wrapper; call predict_action()")

    def _denorm_robot_state(self, robot_state):  # pragma: no cover
        raise NotImplementedError("TwoPhasePolicy is a wrapper; call predict_action()")

    def infer_y(self, obs, robot_state, return_traj: bool):  # pragma: no cover
        raise NotImplementedError("TwoPhasePolicy is a wrapper; call predict_action()")

