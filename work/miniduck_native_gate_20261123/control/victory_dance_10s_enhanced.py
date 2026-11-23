"""Enhanced ten-second victory dance; old VictoryDance10 is unchanged.

The standard template is the validated fallback. This class only builds target
trajectories; the shared mission runner still owns physics, slew limits and
post-dance stability checks. Amplitude is an explicit experiment parameter.
"""
import math
import numpy as np
from control.victory_dance import VictoryDance


class VictoryDance10Enhanced(VictoryDance):
    def __init__(self, default, limits, amplitude=2.0):
        super().__init__(default, limits, duration=10.0, dt=.02)
        self.amplitude = float(amplitude)
        if not (1.0 <= self.amplitude <= 3.0):
            raise ValueError("amplitude must be in [1,3]")

    def _offset(self, t):
        a = self.amplitude
        off = np.zeros(10)
        if t < 1.2:
            # visible crouch then rise
            u = math.sin(math.pi * t / 1.2)
            off[2] = off[7] = .085 * a * u
            off[3] = off[8] = -.13 * a * u
            off[4] = off[9] = .055 * a * u
            phase = 'crouch_rise'
        elif t < 3.2:
            # weight shift and body sway: mirrored hip roll/yaw signs
            u = math.sin(2 * math.pi * (t - 1.2) / 1.0)
            off[0] = .070 * a * u
            off[5] = -.070 * a * u
            off[1] = .065 * a * u
            off[6] = -.065 * a * u
            off[4] = .030 * a * u
            off[9] = -.030 * a * u
            phase = 'weight_sway'
        elif t < 6.4:
            # alternating steps: establish shift, then lift one foot
            u = math.sin(2 * math.pi * (t - 3.2) / 0.8)
            left = max(0.0, u) ** 2
            right = max(0.0, -u) ** 2
            off[1] = .040 * a * u
            off[6] = -.040 * a * u
            off[2] = .075 * a * left
            off[3] = -.115 * a * left
            off[4] = .045 * a * left
            off[7] = .075 * a * right
            off[8] = -.115 * a * right
            off[9] = .045 * a * right
            phase = 'alternating_steps'
        elif t < 8.8:
            # three excited bounces
            u = math.sin(3 * 2 * math.pi * (t - 6.4) / 2.4)
            off[2] = off[7] = .095 * a * u
            off[3] = off[8] = -.145 * a * abs(u)
            off[4] = off[9] = .055 * a * abs(u)
            phase = 'three_bounces'
        else:
            # victory pose, then smooth recenter
            u = math.sin(math.pi * (t - 8.8) / 1.2)
            off[2] = off[7] = .080 * a * u
            off[3] = off[8] = -.120 * a * u
            off[4] = off[9] = .045 * a * u
            off[0] = .030 * a * u
            off[5] = -.030 * a * u
            phase = 'victory_recenter'
        return off, phase


def validate_candidate(default, limits, amplitude, max_step=5.24 * .02 + 1e-9):
    d = VictoryDance10Enhanced(default, limits, amplitude)
    traj = d.trajectory(); prev = np.asarray(default, dtype=np.float64)
    md = 0.0; phases = []
    for s in traj:
        cur = np.asarray(s.target)
        md = max(md, float(np.max(np.abs(cur - prev))))
        if not s.bounded:
            return {'amplitude': amplitude, 'pass': False, 'reason': 'joint_limit'}
        if not phases or phases[-1] != s.phase:
            phases.append(s.phase)
        prev = cur
    return {'amplitude': amplitude, 'pass': md <= max_step,
            'intervals': len(traj), 'duration_s': d.duration,
            'max_target_delta_rad': md, 'max_allowed_delta_rad': max_step,
            'endpoint_distance_rad': float(np.max(np.abs(np.asarray(traj[-1].target) - d.default))),
            'phases': phases}
