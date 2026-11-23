"""Shared native MuJoCo mission execution chain.

The evaluator and the renderer both call :func:`run_mission`.  The renderer is
only an observer of the exact same control/state sequence; it must not own a
second FSM or a second command profile.

Command sources:
  mission       current MissionFSM (old generic mission command)
  p070          exact R4 p070 segment speed and small lateral guide
  p070_mission  p070 until the final approach, then MissionFSM braking/settle
"""
from dataclasses import dataclass, field
import json
import math
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import acceptance_5m as A
import r4_profile as R
from control.mission_fsm import MissionFSM, MotionCommand
from control.victory_dance import VictoryDance
from control.victory_dance_10s import VictoryDance10
from control.victory_dance_10s_enhanced import VictoryDance10Enhanced
from perception.goal_adapter import Pose2D

DT = 0.02
DEFAULT_PROFILE = dict(flat=.110, entry=.08, exit=.07,
                       uphill=.70, crest=.70, downhill=.70, min=0.)
# Explicit routing; unknown/zero-progress modes must never default to walking.
COMMAND_ROUTES = {
    "brake": "hold", "settle": "hold", "victory": "hold",
    "hold": "hold", "safe_hold": "hold", "re-approach": "hold",
    "walk": "walk", "strafe": "walk", "creep": "walk",
    "turn": "walk", "p070": "walk",
}
def command_route(mode):
    return COMMAND_ROUTES.get(mode, "hold")


@dataclass
class MissionConfig:
    command_mode: str = "mission"       # mission | p070 | p070_mission
    terrain: str = "flat"
    goal_xy: tuple = (5.0, 0.0)
    stop_x: float = 5.02
    max_seconds: float = 140.0
    seed: int | None = None
    start_xyz: tuple | None = None
    arrival_radius_m: float = .10
    settle_s: float = 2.0
    dance_s: float = 5.0
    hold_s: float = 2.0
    profile: dict = field(default_factory=lambda: dict(DEFAULT_PROFILE))
    # None preserves the exact original p070 min(.04, .45*cmd_x) rule.
    guide_cap: float | None = None
    joint_noise: float = .003
    dance: bool = False
    # standard preserves the validated old template; enhanced is an explicit
    # experimental trajectory and never becomes the default.
    dance_style: str = "standard"
    dance_amplitude: float = 1.0
    contract: dict | None = None
    # Explicit opt-in for far-distance flat navigation cruise. None preserves
    # the historical .06 m/s cap; p070 keeps using profile["flat"].
    flat_cruise_m_s: float | None = None
    render_every: int = 2
    trace: bool = True


def p070_command(x, y, cfg):
    """Exact p070 segment command including the original small lateral guide."""
    x = float(x)
    if cfg.terrain == "flat":
        cmd_x = float(cfg.profile["flat"])
    elif x < .95:
        cmd_x = float(cfg.profile["entry"])
    elif x < 2.30:
        cmd_x = max(.05 * float(cfg.profile["uphill"]), float(cfg.profile["min"]))
    elif x < 2.75:
        cmd_x = max(.045 * float(cfg.profile["crest"]), float(cfg.profile["min"]))
    elif x < 4.20:
        cmd_x = max(.04 * float(cfg.profile["downhill"]), float(cfg.profile["min"]))
    else:
        cmd_x = float(cfg.profile["exit"])
    guide_eff = (float(cfg.guide_cap) if cfg.guide_cap is not None
                 else min(.04, .45 * max(cmd_x, 1e-6)))
    vy = float(np.clip(float(cfg.goal_xy[1]) - float(y), -guide_eff, guide_eff))
    return MotionCommand(cmd_x, vy, 0.0, 1.0, "p070")


def _initial_pose(cfg):
    if cfg.start_xyz is not None:
        return tuple(float(v) for v in cfg.start_xyz)
    if cfg.seed is None:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(int(cfg.seed))
    return (float(rng.uniform(-.02, .02)),
            float(rng.uniform(-.04, .04)),
            float(rng.uniform(-.02, .02)))


def _dance_template(cfg):
    if not cfg.dance:
        return None
    if cfg.contract is None:
        raise ValueError("dance=True requires contract JSON")
    if cfg.dance_style == "enhanced":
        if abs(float(cfg.dance_s) - 10.0) > 1e-9:
            raise ValueError("enhanced dance currently requires dance_s=10.0")
        return VictoryDance10Enhanced(cfg.contract["default_joint_positions"],
                                      cfg.contract["joint_limits"],
                                      amplitude=cfg.dance_amplitude)
    if cfg.dance_style != "standard":
        raise ValueError("unknown dance_style: " + str(cfg.dance_style))
    if abs(float(cfg.dance_s) - 10.0) < 1e-9:
        return VictoryDance10(cfg.contract["default_joint_positions"],
                             cfg.contract["joint_limits"])
    return VictoryDance(cfg.contract["default_joint_positions"],
                        cfg.contract["joint_limits"],
                        duration=cfg.dance_s, dt=DT)


def run_mission(policy, adapter, spec, cfg: MissionConfig, observer=None):
    """Run one native mission and return metrics plus optional trace.

    ``observer(event)`` is called after each rendered tick's physics advance.
    It receives the same ``data`` object and command/state used by the run; it
    must not modify control state.
    """
    if cfg.command_mode not in ("mission", "p070", "p070_mission"):
        raise ValueError("unknown command_mode: " + cfg.command_mode)
    R.init_custom(cfg.terrain)
    x0, y0, yaw0 = _initial_pose(cfg)
    A.reset(x0, y0, yaw0, joint_noise=cfg.joint_noise)
    policy.reset()
    cruise = (.06 if cfg.terrain != "flat" or cfg.flat_cruise_m_s is None
              else float(cfg.flat_cruise_m_s))
    fsm = MissionFSM(arrival_radius_m=min(.08, float(spec.arrival_radius_m)),
                     approach_margin_m=min(.08, float(spec.arrival_radius_m)),
                     acceptance_radius_m=float(spec.arrival_radius_m),
                     settle_s=cfg.settle_s, dance_s=cfg.dance_s, hold_s=cfg.hold_s,
                     cruise_speed_m_s=cruise)
    dance = _dance_template(cfg)
    settle_needed = max(1, int(round(cfg.settle_s / DT)))
    dance_needed = dance.intervals if dance is not None else 0
    hold_needed = max(1, int(round(cfg.hold_s / DT))) if dance is not None else 0

    t = 0.0
    reached_p070 = False
    hold_from = None
    hold_start = 0.0
    strict_nav_ticks = 0
    legacy_nav_ticks = 0
    hold_ticks = 0
    dance_idx = 0
    dance_started_at = None
    dance_ended_at = None
    dance_phases = []
    mission_success = False
    fall = False
    fall_reason = None
    wall_touch = False
    body_ground = False
    max_tilt = 0.0
    min_z = 9.0
    max_speed = 0.0
    trace = []
    path = []
    status = fsm._status()
    runtime_state = "NAVIGATE"
    last_applied = np.asarray(A.data.ctrl, dtype=np.float32).copy()
    allowed_delta = float(policy.contract["max_motor_velocity"] * policy.contract["dt"])
    max_slew_delta = 0.0
    slew_violations = 0

    for tick in range(int(round(cfg.max_seconds / DT))):
        t = tick * DT
        pose = Pose2D(float(A.data.qpos[0]), float(A.data.qpos[1]), float(A.heading()))
        est = adapter.estimate(pose, t)
        speed = float(np.linalg.norm(A.data.qvel[:2]))
        max_speed = max(max_speed, speed)
        yaw_rate = float(A.data.qvel[5])
        angular = float(np.linalg.norm(A.data.qvel[3:6]))
        proxy = type("RobotProxy", (), {"speed": speed, "yaw_rate": yaw_rate,
                                        "angular_speed": angular})()
        state_before = fsm.state
        # The policy's target history must describe the last target actually
        # written to data.ctrl, not a desired target that was slew-limited.
        policy.targets = last_applied.copy()

        if cfg.command_mode == "p070":
            if not reached_p070 and float(A.data.qpos[0]) < cfg.stop_x:
                cmd = p070_command(A.data.qpos[0], A.data.qpos[1], cfg)
            else:
                reached_p070 = True
                cmd = MotionCommand(0., 0., 0., 1., "brake")
            status = fsm._status()
        elif cfg.command_mode == "p070_mission":
            if not reached_p070 and float(A.data.qpos[0]) < cfg.stop_x - .10:
                cmd = p070_command(A.data.qpos[0], A.data.qpos[1], cfg)
                status = fsm._status()
            else:
                reached_p070 = True
                cmd, status = fsm.step(est, proxy, DT)
        else:
            cmd, status = fsm.step(est, proxy, DT)

        applying_dance = bool(dance is not None and status.state == "VICTORY"
                              and dance_idx < dance_needed)
        if applying_dance:
            dance_t = min(dance_idx * DT, dance.duration)
            sample = dance.sample(dance_t)
            target = np.clip(np.asarray(sample.target),
                             policy.limits[:, 0], policy.limits[:, 1])
            hold_from = None
            if dance_started_at is None:
                dance_started_at = t
            dance_ended_at = t + DT
            if not dance_phases or dance_phases[-1][0] != sample.phase:
                dance_phases.append([sample.phase, t])
            dance_idx += 1
        elif ((cfg.command_mode == "p070" and reached_p070)
              or command_route(cmd.mode) == "hold"):
            if hold_from is None:
                hold_from = policy.targets.copy()
                hold_start = t
            blend = min(1.0, (t - hold_start) / .45)
            target = np.clip(hold_from + blend * (policy.default - hold_from),
                             policy.limits[:, 0], policy.limits[:, 1])
        else:
            hold_from = None
            obs = policy.build(A.sensor("gyro"), A.sensor("accelerometer"),
                               np.asarray([cmd.vx, cmd.vy, cmd.yaw_rate], np.float32),
                               A.data.qpos[A.qaddr], A.data.qvel[A.daddr],
                               A.contacts(), heading=A.heading())
            raw = policy.infer(obs)
            target = policy.target(raw)
            policy.commit(raw, target)

        desired_target = np.asarray(target, dtype=np.float32)
        previous_applied = last_applied.copy()
        desired_clipped = np.clip(desired_target, policy.limits[:, 0], policy.limits[:, 1])
        if (np.any(previous_applied < policy.limits[:, 0] - 1e-6)
                or np.any(previous_applied > policy.limits[:, 1] + 1e-6)):
            raise ValueError("last_applied target is outside joint limits")
        delta = desired_clipped - previous_applied
        if np.any(np.abs(delta) > allowed_delta + 1e-6):
            applied_target = previous_applied + np.clip(delta, -allowed_delta, allowed_delta)
            slew_reason = "slew_limit"
        else:
            applied_target = desired_clipped
            slew_reason = ("joint_limit"
                           if not np.array_equal(desired_target, desired_clipped) else "none")
        applied_target = np.clip(applied_target, policy.limits[:, 0], policy.limits[:, 1])
        applied_delta = float(np.max(np.abs(applied_target - previous_applied)))
        max_slew_delta = max(max_slew_delta, applied_delta)
        if applied_delta > allowed_delta + 1e-6:
            slew_violations += 1
            raise ValueError(
                f"applied target exceeds slew contract: {applied_delta:.9f} > {allowed_delta:.9f}")
        A.data.ctrl[:] = applied_target
        last_applied = applied_target.copy()
        policy.targets = applied_target.copy()
        for _ in range(10):
            A.mujoco.mj_step(A.model, A.data)
        A.mujoco.mj_forward(A.model, A.data)

        # Stable/settle metrics must use the state after the physics substeps.
        speed = float(np.linalg.norm(A.data.qvel[:2]))
        max_speed = max(max_speed, speed)
        yaw_rate = float(A.data.qvel[5])
        angular = float(np.linalg.norm(A.data.qvel[3:6]))
        r = A.data.site_xmat[A.imu].reshape(3, 3)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))
        max_tilt = max(max_tilt, tilt)
        z = float(A.data.qpos[2])
        min_z = min(min_z, z)
        x = float(A.data.qpos[0])
        y = float(A.data.qpos[1])
        dist = float(math.hypot(x - cfg.goal_xy[0], y - cfg.goal_xy[1]))
        contacts = np.asarray(A.contacts(), dtype=int)
        wall_touch = wall_touch or bool(A.wall_hit())
        body_ground = body_ground or bool(A.body_ground_hit())
        if z < .05:
            fall, fall_reason = True, "root_z<0.05"
        elif tilt > 75.:
            fall, fall_reason = True, "tilt>75"

        stable = bool(est.valid and math.isfinite(float(est.timestamp))
                      and dist <= float(spec.arrival_radius_m)
                      and speed < .02 and abs(yaw_rate) < .10 and angular < .20)
        legacy_stable = bool(est.valid and dist <= .10 and speed < .02)
        strict_nav_ticks = strict_nav_ticks + 1 if stable else 0
        legacy_nav_ticks = legacy_nav_ticks + 1 if legacy_stable else 0

        if dance is not None:
            # The tick that emits the 500th dance interval is still a dance
            # tick. Start the independent post-dance hold count only on the
            # following pure hold tick.
            if dance_idx >= dance_needed and not applying_dance:
                hold_ticks = hold_ticks + 1 if stable else 0
                if hold_ticks >= hold_needed:
                    mission_success = True
        else:
            if strict_nav_ticks >= settle_needed:
                mission_success = True

        if fall or wall_touch or body_ground:
            mission_success = False

        if cfg.trace:
            trace.append([round(t + DT, 3), round(x, 5), round(y, 5), round(z, 5),
                          round(speed, 5), round(float(cmd.vx), 5),
                          round(float(cmd.vy), 5), cmd.mode, status.state,
                          round(tilt, 3), int(contacts[0]), int(contacts[1]),
                          int(strict_nav_ticks), int(legacy_nav_ticks),
                          int(dance_idx), [round(float(v), 6) for v in applied_target],
                          [round(float(v), 6) for v in desired_target],
                          [round(float(v), 6) for v in previous_applied], slew_reason])
        path.append([x, y])
        if observer is not None and tick % cfg.render_every == 0:
            observer({"tick": tick, "t": t + DT, "data": A.data, "model": A.model,
                      "status": status, "cmd": cmd, "target": target,
                      "dance_idx": dance_idx, "dance_phases": dance_phases,
                      "state": status.state if cfg.command_mode != "p070" else
                      ("P070_HOLD" if reached_p070 else "P070_RUN"),
                      "tilt": tilt, "x": x, "y": y, "z": z, "dist": dist,
                      "stable": stable, "strict_nav_ticks": strict_nav_ticks})

        if mission_success or fall or wall_touch or body_ground or status.state == "SAFE_HOLD":
            break

    final_distance = float(math.hypot(float(A.data.qpos[0]) - cfg.goal_xy[0],
                                      float(A.data.qpos[1]) - cfg.goal_xy[1]))
    no_failure = not (fall or wall_touch or body_ground)
    strict_nav_success = bool(no_failure and strict_nav_ticks >= settle_needed
                              and final_distance <= float(spec.arrival_radius_m))
    strict_mission_success = bool(strict_nav_success and
                                  ((dance_idx >= dance_needed and hold_ticks >= hold_needed)
                                   if dance is not None else True))
    legacy_nav_success = bool(no_failure and legacy_nav_ticks >= settle_needed
                              and final_distance <= .10)
    runtime_state = (status.state if cfg.command_mode != "p070" else
                     ("P070_DONE" if strict_nav_success else
                      ("P070_HOLD" if reached_p070 else "P070_RUN")))
    result = {
        "schema": "shared_native_mission_v1",
        "command_mode": cfg.command_mode,
        "terrain": cfg.terrain,
        "goal_xy": [float(v) for v in cfg.goal_xy],
        "start_xyz": [float(v) for v in _initial_pose(cfg)],
        "seed": cfg.seed,
        "profile": dict(cfg.profile),
        "seconds": t + DT,
        "state": runtime_state,
        "final_distance_m": final_distance,
        "max_tilt_deg": max_tilt,
        "max_speed_m_s": max_speed,
        "min_root_z": min_z,
        "fall_reason": fall_reason,
        "fell": bool(fall),
        "wall_hit": bool(wall_touch),
        "body_ground_hit": bool(body_ground),
        "nav_success": bool(strict_nav_success),
        "legacy_nav_success": legacy_nav_success,
        "dance_success": bool(dance is not None and dance_idx >= dance_needed),
        "mission_success": bool(strict_mission_success),
        "strict_nav_success": strict_nav_success,
        "strict_mission_success": strict_mission_success,
        "settle_ticks_required": settle_needed,
        "hold_ticks_required": hold_needed,
        "dance_steps": int(dance_idx),
        "dance_expected_steps": int(dance_needed),
        "dance_started_at_s": dance_started_at,
        "dance_ended_at_s": dance_ended_at,
        "dance_phases": dance_phases,
        "slew_contract": {
            "allowed_rad_per_tick": allowed_delta,
            "max_abs_delta_rad": max_slew_delta,
            "violations": int(slew_violations),
            "pass": bool(slew_violations == 0),
        },
        "metadata": {
            "python": sys.executable,
            "mujoco": getattr(A.mujoco, "__version__", None),
            "solver": int(A.model.opt.solver),
            "integrator": int(A.model.opt.integrator),
            "timestep": float(A.model.opt.timestep),
            "hardware": False,
            "perception": "explicit_coordinate_or_simulated_oracle",
        },
        "trace": trace if cfg.trace else [],
        "path": path,
    }
    return result


def load_contract(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
