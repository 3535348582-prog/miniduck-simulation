"""Deterministic five-second leg-only victory dance, bounded by joint limits."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class DanceSample:
    t: float
    target: tuple
    phase: str
    bounded: bool


class VictoryDance:
    def __init__(self, default, limits, duration=5.0, dt=0.02):
        self.default = np.asarray(default, dtype=np.float64).reshape(10)
        self.limits = np.asarray(limits, dtype=np.float64).reshape(10, 2)
        self.duration = float(duration)
        self.dt = float(dt)
        self.intervals = int(round(self.duration / self.dt))

    def _offset(self, t):
        off = np.zeros(10)
        if t < 1.0:
            u = math.sin(math.pi * t / 1.0)
            off[2] = off[7] = .06 * u
            off[3] = off[8] = -.09 * u
            off[4] = off[9] = .04 * u
            phase = 'crouch_rise'
        elif t < 2.5:
            u = math.sin(2 * math.pi * (t - 1.0) / 1.5)
            off[1] = off[6] = .05 * u
            off[4] = off[9] = .025 * u
            phase = 'wiggle'
        elif t < 4.0:
            u = math.sin(2 * math.pi * (t - 2.5) / .75)
            off[2] = off[7] = .045 * u
            off[3] = off[8] = -.07 * abs(u)
            phase = 'bounce'
        else:
            u = math.sin(math.pi * (t - 4.0) / 1.0)
            off[2] = off[7] = .020 * u
            off[3] = off[8] = -.030 * u
            phase = 'recenter_pose'
        return off, phase

    def sample(self, t):
        if t < 0 or t > self.duration:
            raise ValueError('dance sample outside duration')
        off, phase = self._offset(float(t))
        target = np.clip(self.default + off, self.limits[:, 0], self.limits[:, 1])
        return DanceSample(float(t), tuple(float(x) for x in target), phase,
                           bool(np.all(target >= self.limits[:, 0]) and np.all(target <= self.limits[:, 1])))

    def trajectory(self):
        return [self.sample(i * self.dt) for i in range(self.intervals)]

    def validate(self, max_step=5.24 * .02 + 1e-9):
        traj = self.trajectory()
        prev = self.default.copy()
        max_delta = 0.0
        for s in traj:
            cur = np.asarray(s.target)
            max_delta = max(max_delta, float(np.max(np.abs(cur - prev))))
            if not s.bounded:
                return {'pass': False, 'reason': 'joint_limit'}
            prev = cur
        return {'pass': bool(max_delta <= max_step and len(traj) == 250),
                'intervals': len(traj), 'duration_s': self.duration,
                'max_target_delta_rad': max_delta, 'max_allowed_delta_rad': max_step,
                'endpoint_distance_rad': float(np.max(np.abs(np.asarray(traj[-1].target) - self.default)))}
