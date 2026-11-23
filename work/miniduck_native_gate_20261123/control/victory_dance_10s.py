"""Ten-second victory dance: 500 control intervals at 50 Hz."""
import math
import numpy as np
from control.victory_dance import VictoryDance


class VictoryDance10(VictoryDance):
    def __init__(self, default, limits):
        super().__init__(default, limits, duration=10.0, dt=.02)

    def _offset(self, t):
        off = np.zeros(10)
        if t < 1.0:
            u = math.sin(math.pi * t / 1.0)
            off[2] = off[7] = .06 * u
            off[3] = off[8] = -.09 * u
            off[4] = off[9] = .04 * u
            phase = 'crouch_rise'
        elif t < 3.0:
            u = math.sin(2 * math.pi * (t - 1.0) / 1.0)
            off[1] = off[6] = .05 * u
            off[4] = off[9] = .025 * u
            phase = 'wiggle'
        elif t < 6.0:
            u = math.sin(2 * math.pi * (t - 3.0) / 1.0)
            off[2] = off[7] = .045 * u
            off[3] = off[8] = -.07 * abs(u)
            phase = 'bounce'
        elif t < 8.0:
            u = math.sin(2 * math.pi * (t - 6.0) / 1.0)
            off[1] = off[6] = .04 * u
            off[4] = off[9] = .02 * u
            phase = 'step_sway'
        else:
            u = math.sin(math.pi * (t - 8.0) / 2.0)
            off[2] = off[7] = .02 * u
            off[3] = off[8] = -.03 * u
            phase = 'victory_hold'
        return off, phase

    def validate(self, max_step=5.24 * .02 + 1e-9):
        traj = self.trajectory()
        prev = self.default.copy()
        max_delta = 0.0
        phases = []
        for s in traj:
            cur = np.asarray(s.target)
            max_delta = max(max_delta, float(np.max(np.abs(cur - prev))))
            if not s.bounded:
                return {'pass': False, 'reason': 'joint_limit'}
            if not phases or phases[-1] != s.phase:
                phases.append(s.phase)
            prev = cur
        return {'pass': bool(max_delta <= max_step and len(traj) == 500),
                'intervals': len(traj), 'duration_s': self.duration,
                'max_target_delta_rad': max_delta,
                'max_allowed_delta_rad': max_step,
                'endpoint_distance_rad': float(np.max(np.abs(np.asarray(traj[-1].target) - self.default))),
                'phases': phases}
