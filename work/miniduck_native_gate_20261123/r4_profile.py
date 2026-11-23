#!/usr/bin/env python3
"""R4 segment-aware 5 m profile runner.

Keeps policy weights and the 40-step phase unchanged. It changes only the
commanded speed profile by terrain segment and records physical diagnostics.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import mujoco
import acceptance_5m as A
import r4_terrain
from rebuilt_policy import RebuiltHeadingPolicy



def init_custom(terrain):
    deg = {"flat": 0, "slope5": 5, "slope6": 6, "slope65": 6.5, "slope7": 7, "slope75": 7.5, "slope8": 8, "slope10": 10}[terrain]
    A.TERRAIN = terrain
    A.TERRAIN_DEG = deg
    A.SLOPE_SCALE = 1.0
    A.SLOPE_MIN = 0.0
    A.GUIDE_RATIO = .45
    A.terrain_course = r4_terrain
    A.model = r4_terrain.model_for_acceptance(deg, walls=True)
    A.ramp_names = [] if float(deg) == 0 else ["up_ramp", "top_flat", "down_ramp"]
    A.data = A.mujoco.MjData(A.model)
    A.imu = A.model.site("imu").id
    A.floor = A.model.geom("floor").id
    A.ground = [A.floor] + [A.model.geom(n).id for n in A.ramp_names]
    A.feet = [A.model.geom(n).id for n in ("left_foot_bottom_tpu", "right_foot_bottom_tpu")]
    A.walls = {A.model.geom(n).id for n in ("wall_left", "wall_right")}
    A.robot_bodies = {i for i in range(A.model.nbody) if A.model.body(i).name not in ("world", "floor")}
    A.robot_geoms = {gi for gi in range(A.model.ngeom) if int(A.model.geom_bodyid[gi]) in A.robot_bodies}
    A.qaddr = A.model.jnt_qposadr[A.model.actuator_trnid[:, 0]]
    A.daddr = A.model.jnt_dofadr[A.model.actuator_trnid[:, 0]]

DT = .02
MAX_MOTOR_VELOCITY = 5.24
TARGET = np.array([A.TARGET_X, A.TARGET_Y], float)


def segment_for(x):
    if A.TERRAIN == "flat":
        return "flat"
    if x < .95:
        return "pre"
    if x < 2.30:
        return "uphill"
    if x < 2.75:
        return "crest"
    if x < 4.20:
        return "downhill"
    return "post"


def profile_speed(x, p):
    if A.TERRAIN == "flat":
        return float(p["flat"])
    if x < .95:
        return float(p["entry"])
    if x < 2.30:
        return max(.05 * p["uphill"], p["min"])
    if x < 2.75:
        return max(.045 * p["crest"], p["min"])
    if x < 4.20:
        return max(.04 * p["downhill"], p["min"])
    return float(p["exit"])


def foot_bottom_z(gid):
    m, d = A.model, A.data
    mesh = m.geom_dataid[gid]
    start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    rot = d.geom_xmat[gid].reshape(3, 3)
    return float(d.geom_xpos[gid, 2] + np.min(m.mesh_vert[start:start + count] @ rot[2]))


def force_limit():
    try:
        return np.maximum(np.abs(np.asarray(A.model.actuator_forcerange)[:, 1]), 1e-6)
    except Exception:
        return np.ones(A.model.nu) * 3.23


def run_trial(policy, seed, p, stop_x=5.02, max_ticks=7000):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-.02, .02)
    y0 = rng.uniform(-.04, .04)
    yaw0 = rng.uniform(-.02, .02)
    A.reset(x0, y0, yaw0, joint_noise=.003)
    policy.reset()
    x_start = float(A.data.qpos[0])
    reached_at = None
    settled_since = None
    settled_seconds = 0.
    fall = False
    fall_reason = None
    termination_reason = "max_steps"
    holding = False
    hold_tick = None
    hold_from = None
    wall_touch = False
    body_ground = False
    max_tilt = 0.
    min_z = 9.
    max_angvel = 0.
    prev_x = x_start
    prev_contacts = np.asarray(A.contacts(), dtype=int)
    prev_foot_xy = np.asarray(A.data.geom_xpos[A.feet, :2], dtype=float)
    last_strike_x = {"left": None, "right": None}
    segments = {}
    trace = []

    def seg(name):
        if name not in segments:
            segments[name] = dict(duration=0., dx=0., speed_sum=0., ticks=0, steps=0,
                                  step_lengths=[], double_support=0, swing_clearance=[],
                                  contact_slip=[], angvel_max=0., joint_v_max=0.,
                                  joint_t_max=0., joint_v_sat=0., joint_t_sat=0.,
                                  contact_ticks=0)
        return segments[name]

    for tick in range(max_ticks):
        x, y = float(A.data.qpos[0]), float(A.data.qpos[1])
        dist = math.hypot(x - TARGET[0], y - TARGET[1])
        if reached_at is None:
            if x >= stop_x:
                reached_at = tick * DT
                holding = True
                hold_tick = tick
                hold_from = policy.targets.copy()
                cmd = np.zeros(3, np.float32)
            else:
                ey = TARGET[1] - y
                cmd_x = profile_speed(x, p)
                guide_eff = min(.04, .45 * max(cmd_x, 1e-6))
                vy = float(np.clip(ey, -guide_eff, guide_eff))
                cmd = np.array([cmd_x, vy, 0.], np.float32)
        else:
            cmd = np.zeros(3, np.float32)

        if holding:
            blend = min(1.0, (tick - hold_tick) * DT / .45)
            target = np.clip(hold_from + blend * (policy.default - hold_from),
                             policy.limits[:, 0], policy.limits[:, 1])
        else:
            obs = policy.build(A.sensor("gyro"), A.sensor("accelerometer"), cmd,
                               A.data.qpos[A.qaddr], A.data.qvel[A.daddr],
                               A.contacts(), heading=A.heading())
            raw = policy.infer(obs)
            target = policy.target(raw)
            policy.commit(raw, target)

        A.data.ctrl[:] = target
        for _ in range(10):
            mujoco.mj_step(A.model, A.data)
        mujoco.mj_forward(A.model, A.data)

        x, y = float(A.data.qpos[0]), float(A.data.qpos[1])
        name = segment_for(x)
        s = seg(name)
        s["duration"] += DT
        s["ticks"] += 1
        s["dx"] += x - prev_x
        speed = math.hypot(float(A.data.qvel[0]), float(A.data.qvel[1]))
        s["speed_sum"] += speed
        r = A.data.site_xmat[A.imu].reshape(3, 3)
        tilt = math.degrees(math.acos(max(-1., min(1., float(r[2, 2])))))
        max_tilt = max(max_tilt, tilt)
        min_z = min(min_z, float(A.data.qpos[2]))
        angvel = float(np.linalg.norm(np.asarray(A.data.qvel[3:6])))
        max_angvel = max(max_angvel, angvel)
        s["angvel_max"] = max(s["angvel_max"], angvel)
        c = np.asarray(A.contacts(), dtype=int)
        foot_z = np.array([foot_bottom_z(g) for g in A.feet])
        foot_xy = np.asarray(A.data.geom_xpos[A.feet, :2], dtype=float)
        foot_slip = np.linalg.norm(foot_xy - prev_foot_xy, axis=1) / DT
        qv = np.abs(np.asarray(A.data.qvel[A.daddr]))
        tf = np.abs(np.asarray(A.data.actuator_force))
        fl = force_limit()
        s["joint_v_max"] = max(s["joint_v_max"], float(np.max(qv)))
        s["joint_t_max"] = max(s["joint_t_max"], float(np.max(tf)))
        s["joint_v_sat"] += int(np.any(qv >= .99 * MAX_MOTOR_VELOCITY))
        s["joint_t_sat"] += int(np.any(tf >= .99 * fl))
        s["contact_ticks"] += int(np.any(c))
        if c[0] and c[1]:
            s["double_support"] += 1
        for i, key in enumerate(("left", "right")):
            if c[i] and not prev_contacts[i]:
                s["steps"] += 1
                if last_strike_x[key] is not None:
                    s["step_lengths"].append(float(x - last_strike_x[key]))
                last_strike_x[key] = x
            if not c[i]:
                s["swing_clearance"].append(float(foot_z[i]))
            elif prev_contacts[i]:
                s["contact_slip"].append(float(foot_slip[i]))

        trace.append([round(tick * DT, 3), round(x, 5), round(y, 5), round(float(A.data.qpos[2]), 5), round(speed, 5), round(float(cmd[0]), 5), round(float(cmd[1]), 5), round(tilt, 3), int(c[0]), int(c[1]), round(float(foot_z[0]), 5), round(float(foot_z[1]), 5)])
        if A.data.qpos[2] < .05:
            fall = True
            fall_reason = "root_z<0.05"
        elif tilt > 75.:
            fall = True
            fall_reason = "tilt>75"
        if A.wall_hit():
            wall_touch = True
        if A.body_ground_hit():
            body_ground = True
        if reached_at is not None:
            if dist <= .10 and speed < .02:
                if settled_since is None:
                    settled_since = tick * DT
                settled_seconds = tick * DT - settled_since
            else:
                settled_since = None
                settled_seconds = 0.
        prev_x, prev_contacts, prev_foot_xy = x, c, foot_xy
        if fall or wall_touch or body_ground or settled_seconds >= 2.0:
            termination_reason = (fall_reason if fall else
                                  "wall_hit" if wall_touch else
                                  "body_ground_hit" if body_ground else "settled")
            break

    elapsed = (tick + 1) * DT
    final_dist = float(math.hypot(A.data.qpos[0] - TARGET[0], A.data.qpos[1] - TARGET[1]))
    success = (reached_at is not None and final_dist <= .10 and settled_seconds >= 2.0
               and not fall and not wall_touch and not body_ground)
    for s in segments.values():
        n = max(s["ticks"], 1)
        s["avg_speed_m_s"] = s["dx"] / max(s["duration"], 1e-9)
        s["mean_speed_m_s"] = s["speed_sum"] / n
        s["cadence_hz"] = s["steps"] / max(s["duration"], 1e-9)
        s["double_support_fraction"] = s["double_support"] / n
        s["contact_fraction"] = s["contact_ticks"] / n
        s["step_length_mean_m"] = float(np.mean(s["step_lengths"])) if s["step_lengths"] else None
        s["swing_clearance_max_m"] = max(s["swing_clearance"]) if s["swing_clearance"] else None
        s["contact_slip_mean_m_s"] = float(np.mean(s["contact_slip"])) if s["contact_slip"] else None
        s["contact_slip_max_m_s"] = max(s["contact_slip"]) if s["contact_slip"] else None
        s["joint_v_max_ratio"] = s["joint_v_max"] / MAX_MOTOR_VELOCITY
        s["joint_v_sat_fraction"] = s["joint_v_sat"] / n
        s["joint_t_sat_fraction"] = s["joint_t_sat"] / n
        s["step_lengths"] = [round(x, 5) for x in s["step_lengths"]]
        s["swing_clearance"] = []
        s["contact_slip"] = []

    return dict(seed=int(seed), success=bool(success), fell=bool(fall),
                fall_reason=fall_reason, termination_reason=termination_reason,
                reached_at_s=reached_at, final_distance_m=final_dist,
                settled_seconds=settled_seconds, elapsed_s=elapsed,
                avg_speed_m_s=(float(A.data.qpos[0]) - x_start) / max(elapsed, 1e-9),
                max_tilt_deg=max_tilt, min_root_z=min_z, max_angvel_rad_s=max_angvel,
                wall_hit=wall_touch, body_ground_hit=body_ground, segments=segments, trace=trace)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--terrain", choices=["flat", "slope5", "slope6", "slope65", "slope7", "slope75", "slope8", "slope10"], default="slope6")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed-base", type=int, default=98100)
    ap.add_argument("--max-seconds", type=float, default=140.0)
    ap.add_argument("--stop-x", type=float, default=5.02)
    ap.add_argument("--flat-speed", type=float, default=.10)
    ap.add_argument("--entry-speed", type=float, default=.09)
    ap.add_argument("--exit-speed", type=float, default=.09)
    ap.add_argument("--uphill-scale", type=float, default=.70)
    ap.add_argument("--crest-scale", type=float, default=.70)
    ap.add_argument("--downhill-scale", type=float, default=.70)
    ap.add_argument("--tag", default="r4")
    ap.add_argument("--policy", default=A.DEMO + "/models/duck_initial/policy.onnx")
    ap.add_argument("--rebuilt-params", default=A.DEMO + "/models/duck_initial/slope6_ppo_p3_128k.npz")
    args = ap.parse_args()
    init_custom(args.terrain)
    profile = dict(flat=args.flat_speed, entry=args.entry_speed, exit=args.exit_speed,
                   uphill=args.uphill_scale, crest=args.crest_scale,
                   downhill=args.downhill_scale, min=0.)
    policy = RebuiltHeadingPolicy.load(args.rebuilt_params, args.policy.rsplit(".", 1)[0] + ".json")
    policy.warmup()
    results = [run_trial(policy, args.seed_base + i, profile, args.stop_x,
                         int(round(args.max_seconds / DT))) for i in range(args.trials)]
    out = Path(A.DEMO) / "out" / f"r4_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(tag=args.tag, terrain=args.terrain, profile=profile,
                   max_seconds=args.max_seconds, policy=args.rebuilt_params,
                   success_count=sum(r["success"] for r in results),
                   trial_count=len(results), results=results)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("tag", "terrain", "profile", "success_count", "trial_count")}, indent=2), flush=True)
    for r in results:
        print(json.dumps({k: r[k] for k in ("seed", "success", "fell", "termination_reason",
            "reached_at_s", "final_distance_m", "settled_seconds", "avg_speed_m_s")}, indent=2), flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
