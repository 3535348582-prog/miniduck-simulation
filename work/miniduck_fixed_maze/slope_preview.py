"""Native balance preview for a straight slope, preserving joint and slew limits."""

import copy
import math

from evaluate import A, mujoco, np, wrap
from slope10_predictive_snapshots_20261123 import recovery_choices


PATTERNS = np.asarray([
    [0,0,1,0,0,0,0,-1,0,0],
    [0,0,0,1,0,0,0,0,1,0],
    [0,0,0,0,1,0,0,0,0,1],
    [0,0,1,1,0,0,0,-1,1,0],
    [0,0,0,1,-1,0,0,0,1,-1],
    [0,1,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,-1,0],
    [0,0,0,0,1,0,0,0,0,-1],
    [1,0,0,0,0,1,0,0,0,0],
    [1,0,0,0,0,-1,0,0,0,0],
    [0,1,0,0,0,0,-1,0,0,0],
], dtype=np.float32)
STATE = mujoco.mjtState.mjSTATE_INTEGRATION


class BalancePreview:
    def __init__(self, actor, navigator):
        self.actor = actor
        self.nav = navigator
        self.baseline_infer = actor.infer
        self.data = None
        self.state = None
        self.remaining = 0
        self.residual = np.zeros(10, dtype=np.float32)
        self.phase_rate = 1.
        self.sequence_mean = np.zeros((6, len(PATTERNS)), dtype=np.float32)
        self.rng = np.random.default_rng(4732)
        self.sequence_search = False
        self.applied = 0
        self.forecasts = 0
        self.candidate_evaluations = 0
        self.extended = 0
        self.event_log = []

    def initialize(self):
        self.data = mujoco.MjData(A.model)
        self.state = np.empty(mujoco.mj_stateSize(A.model, STATE))
        self.sensors = [A.model.sensor(name) for name in ("gyro", "accelerometer")]
        self.support = set(A.ground)
        self.feet = {geom: i for i, geom in enumerate(A.feet)}

    def evaluate(self, command, reference, residual, horizon, rate=1., follow=0., steady_rate=1.):
        data = self.data
        mujoco.mj_setState(A.model, data, self.state, STATE)
        mujoco.mj_forward(A.model, data)
        branch = copy.copy(self.actor)
        cls = type(branch)
        baseline_xy = data.qpos[:2].copy()
        c, s = math.cos(reference), math.sin(reference)
        worst = 0.
        risk = 0.
        forward_limit = getattr(self.nav, "preview_forward_limit", .22)
        progress_weight = getattr(self.nav, "preview_progress_weight", 8.)
        for tick in range(horizon):
            rot = data.site_xmat[A.imu].reshape(3, 3)
            heading = math.atan2(rot[1, 0], rot[0, 0])
            tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1, 1))))
            roll = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
            pitch = math.degrees(math.atan2(-rot[2, 0], rot[2, 2]))
            forward_speed = c * data.qvel[0] + s * data.qvel[1]
            lateral_speed = -s * data.qvel[0] + c * data.qvel[1]
            unsupported_contact = False
            contacts = np.zeros(2, dtype=np.float32)
            for contact in data.contact:
                a, b = int(contact.geom1), int(contact.geom2)
                if (a in A.walls and b in A.robot_geoms) or (b in A.walls and a in A.robot_geoms):
                    unsupported_contact = True
                robot = b if a in self.support else a if b in self.support else -1
                if robot in self.feet and contact.dist <= 0.:
                    contacts[self.feet[robot]] = 1.
                elif robot in A.robot_geoms:
                    unsupported_contact = True
            if unsupported_contact or tilt > 68. or data.qpos[2] < .056:
                return 20. - tick / horizon, 90.
            worst = max(worst, tilt)
            risk += (max(0., tilt - 37.) / 17.) ** 2
            risk += .45 * (max(0., abs(roll) - 15.) / 23.) ** 2
            risk += .28 * (max(0., abs(lateral_speed) - .18) / .22) ** 2
            risk += .42 * (max(0., -forward_speed - .05) / .20) ** 2
            if forward_limit == .22:
                risk += .42 * (max(0., forward_speed - forward_limit) / .20) ** 2
            else:
                risk += .70 * (max(0., forward_speed - forward_limit) / .10) ** 2
            risk += .50 * (max(0., abs(wrap(heading - reference)) - .12) / .52) ** 2
            risk += .55 * (max(11. - pitch, pitch - 36., 0.) / 24.) ** 2
            gyro, accel = (data.sensordata[spec.adr[0]:spec.adr[0] + spec.dim[0]] for spec in self.sensors)
            branch.targets = np.asarray(data.ctrl, dtype=np.float32)
            obs = cls.build(branch, gyro, accel, command, data.qpos[A.qaddr],
                            data.qvel[A.daddr], contacts, heading=heading)
            error = float(np.clip(wrap(heading - reference), -.40, .40))
            obs[-2:] = [math.sin(error), math.cos(error)]
            raw = self.baseline_infer(obs)
            if residual.ndim == 2:
                raw = np.clip(raw + residual[min(tick // 6, len(residual) - 1)], -1, 1)
            elif tick < 7:
                raw = np.clip(raw + residual, -1, 1)
            elif tick < 14:
                raw = np.clip(raw + follow, -1, 1)
            target = branch.target(raw)
            phase = branch.phase
            cls.commit(branch, raw, target)
            if tick < 7:
                branch.phase = (phase + rate) % branch.contract["phase_period_steps"]
            elif steady_rate != 1.:
                branch.phase = (phase + steady_rate) % branch.contract["phase_period_steps"]
            data.ctrl[:] = target
            mujoco.mj_step(A.model, data, nstep=10)
            mujoco.mj_forward(A.model, data)
        dx, dy = data.qpos[:2] - baseline_xy
        ahead = c * dx + s * dy
        side = -s * dx + c * dy
        action_size = float(np.linalg.norm(residual)) / max(1, math.sqrt(residual.shape[0]) if residual.ndim == 2 else 1)
        return (risk / horizon + 1.5 * abs(side) - progress_weight * ahead
                + .08 * float(np.linalg.norm(data.qvel[3:5]))
                + .13 * action_size), worst

    def search_sequence(self, command, reference, initial_score):
        shape = self.sequence_mean.shape
        mean = self.sequence_mean.copy()
        noise = .23
        best_control = np.zeros(shape, dtype=np.float32)
        best = initial_score
        for count in (22, 16):
            schedules = self.rng.normal(mean, noise, (count, *shape)).astype(np.float32)
            schedules[0] = mean
            schedules[1] = best_control
            ranked = []
            for schedule in schedules:
                action = np.clip(schedule @ PATTERNS, -.52, .52)
                score, _ = self.evaluate(command, reference, action, 36)
                self.candidate_evaluations += 1
                ranked.append((score, schedule))
                if score < best:
                    best, best_control = score, schedule.copy()
            elite = sorted(ranked, key=lambda item: item[0])[:5]
            mean = np.mean([schedule for _, schedule in elite], axis=0)
            noise *= .62
        self.sequence_mean[:-1] = best_control[1:]
        self.sequence_mean[-1] = 0.
        return np.clip(best_control[0] @ PATTERNS, -.52, .52), best

    def active_on_grade(self):
        threshold = getattr(self.nav, "preview_grade_threshold", 5.8)
        return (self.nav.grade_direction is not None and self.nav.grade_direction < 0
                and max(abs(self.nav.grade), self.nav.approach_grade) > threshold)

    def __call__(self, obs):
        raw = self.baseline_infer(obs)
        reference = self.nav.heading_reference
        if not self.active_on_grade() or reference is None or obs[6] <= .02 or abs(obs[8]) > .005:
            self.remaining = 0
            self.residual.fill(0.)
            self.phase_rate = 1.
            return raw
        if self.state is None:
            self.initialize()
        if self.remaining <= 0:
            mujoco.mj_getState(A.model, A.data, self.state, STATE)
            command = obs[6:9]
            nominal = np.zeros(10, dtype=np.float32)
            horizon = 35
            best, tilt = self.evaluate(command, reference, nominal, horizon)
            self.forecasts += 1
            self.residual = nominal
            self.phase_rate = 1.
            if self.sequence_search and (tilt >= 36. or best > -.28):
                self.residual, best = self.search_sequence(command, reference, best)
            elif not self.sequence_search and (tilt >= 38. or best > -.20):
                self.extended += 1
                controls = [scale * pattern for pattern in PATTERNS for scale in (-.24, .24)]
                if tilt >= 54.:
                    controls.extend(delta for _, delta in recovery_choices())
                ranked = [(best, nominal)]
                for perturbation in controls:
                    score, _ = self.evaluate(command, reference, perturbation, horizon)
                    self.candidate_evaluations += 1
                    ranked.append((score, perturbation))
                    if score < best - .008:
                        self.residual, best = perturbation, score
                strongest = sorted(ranked, key=lambda item: item[0])[:3]
                for _, perturbation in strongest:
                    for phase_rate in (.78, 1.24):
                        score, _ = self.evaluate(command, reference, perturbation, horizon, phase_rate)
                        self.candidate_evaluations += 1
                        if score < best - .008:
                            self.residual, best, self.phase_rate = perturbation, score, phase_rate
                if best >= 3.:
                    for _, perturbation in strongest:
                        for follow in controls[:12]:
                            score, _ = self.evaluate(command, reference, perturbation, horizon, follow=follow)
                            self.candidate_evaluations += 1
                            if score < best - .008:
                                self.residual, best, self.phase_rate = perturbation, score, 1.
                if np.any(self.residual):
                    self.event_log.append((float(A.data.time), float(A.data.qpos[0]), tilt, best))
            self.remaining = 6 if self.sequence_search else 7
        self.remaining -= 1
        self.applied += bool(np.any(self.residual))
        self.nav.phase_rate = self.phase_rate
        return np.clip(raw + self.residual, -1, 1)
