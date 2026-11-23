#!/usr/bin/env python3
"""Continuous 5 m flat-track acceptance for the original Duck baseline.

Physical scene: long floor, two side walls with collision, start at x=0,
finish zone centered x=5.0. Metrics come from MuJoCo qpos/qvel, never from the
command or camera. No episode reset during a trial.
"""
import argparse
import json
import math
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "glfw")
import numpy as np
import mujoco

from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[2]
ROOT = os.environ.get("MINIDUCK_ASSET_ROOT", str(BUNDLE / "miniducktraining_classroom"))
TRAIN = os.environ.get("MINIDUCK_TRAINING", ROOT)
DEMO = os.environ.get("MINIDUCK_ROOT", str(Path(__file__).resolve().parent))
sys.path[:0] = [TRAIN + "/training", ROOT + "/training", ROOT + "/vendor", ROOT + "/tools", ROOT]
from portable_policy import policy_type
from rebuilt_policy import RebuiltHeadingPolicy
import terrain_course

os.chdir(ROOT)
adapter = policy_type()
TARGET_X, TARGET_Y = 5.0, 0.0


def build_scene(terrain="flat"):
    deg = {"flat": 0, "slope5": 5, "slope6": 6, "slope8": 8, "slope10": 10}[terrain]
    model = terrain_course.model_for_acceptance(deg, walls=True)
    ramp_names = [] if deg == 0 else ["up_ramp", "top_flat", "down_ramp"]
    return model, ramp_names

def init_model(terrain):
    global model, ramp_names, data, imu, floor, ground, feet, walls, TERRAIN, SLOPE_SCALE, GUIDE_RATIO, TERRAIN_DEG, SLOPE_MIN
    global robot_bodies, robot_geoms, qaddr, daddr
    TERRAIN = terrain
    TERRAIN_DEG = {"flat": 0, "slope5": 5, "slope6": 6, "slope8": 8, "slope10": 10}[terrain]
    SLOPE_SCALE = float(os.environ.get("DUCK_SLOPE_SCALE", "1"))
    SLOPE_MIN = float(os.environ.get("DUCK_SLOPE_MIN", "0"))
    GUIDE_RATIO = float(os.environ.get("DUCK_GUIDE_RATIO", "0.45"))
    model, ramp_names = build_scene(terrain)
    data = mujoco.MjData(model)
    imu = model.site("imu").id
    floor = model.geom("floor").id
    ground = [floor] + [model.geom(n).id for n in ramp_names]
    feet = [model.geom(n).id for n in ("left_foot_bottom_tpu", "right_foot_bottom_tpu")]
    walls = {model.geom(n).id for n in ("wall_left", "wall_right")}
    robot_bodies = {i for i in range(model.nbody) if model.body(i).name not in ("world", "floor")}
    robot_geoms = {gi for gi in range(model.ngeom) if int(model.geom_bodyid[gi]) in robot_bodies}
    qaddr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    daddr = model.jnt_dofadr[model.actuator_trnid[:, 0]]


init_model("flat")


def walk_speed(x):
    if TERRAIN == "flat":
        return float(os.environ.get("DUCK_FLAT_SPEED", ".09"))
    base = float(np.interp(x, [0., .55, .95, 2.30, 2.35, 2.70, 2.75, 4.20, 4.55, 4.90, 5.10],
                           [.08, .08, .05, .05, .045, .045, .04, .04, .045, .065, .07]))
    return max(base * SLOPE_SCALE, SLOPE_MIN)


def rotate_yaw(quat, yaw):
    qy = np.zeros(4)
    mujoco.mju_axisAngle2Quat(qy, np.array([0., 0., 1.]), yaw)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qy, quat)
    return out


def reset(x0=0., y0=0., yaw0=0., joint_noise=0.):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[0] = x0
    data.qpos[1] = y0
    data.qpos[3:7] = rotate_yaw(data.qpos[3:7].copy(), yaw0)
    if joint_noise:
        data.qpos[7:] += np.random.default_rng(1234).normal(0., joint_noise, 10)
        data.qpos[7:] = np.clip(data.qpos[7:], model.jnt_range[1:, 0], model.jnt_range[1:, 1])
    mujoco.mj_forward(model, data)
    verts = []
    for geom in feet:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        rot = data.geom_xmat[geom].reshape(3, 3)
        verts.append(data.geom_xpos[geom, 2] + np.min(model.mesh_vert[start:start + count] @ rot[2]))
    data.qpos[2] += terrain_course.height_np(x0, TERRAIN_DEG) + .0005 - min(verts)
    mujoco.mj_forward(model, data)


def sensor(name):
    s = model.sensor(name)
    return data.sensordata[s.adr[0]:s.adr[0] + s.dim[0]].copy()


def contacts():
    out = np.zeros(2, np.float32)
    for i, g in enumerate(feet):
        out[i] = any(set(c.geom) == {g, gg} and c.dist <= 0
                     for c in data.contact for gg in ground)
    return out


def heading():
    r = data.site_xmat[imu].reshape(3, 3)
    return float(np.arctan2(r[1, 0], r[0, 0]))


def wall_hit():
    return any(({int(c.geom1), int(c.geom2)} & walls) and
               ({int(c.geom1), int(c.geom2)} & robot_geoms)
               for c in data.contact)


def body_ground_hit():
    for c in data.contact:
        pair = {int(c.geom1), int(c.geom2)}
        if (pair & set(ground)) and (pair - set(ground)) & robot_geoms:
            other = (pair - set(ground)).pop()
            if other not in feet:
                return True
    return False


def run_trial(policy, seed, video_path=None, stop_x=5.02, guide_gain=1.0, guide_max=.04,
              hold_seconds=.45, celebrate=False, start_x=None, max_ticks=7000):
    rng = np.random.default_rng(seed)
    x0 = float(start_x) if start_x is not None else rng.uniform(-.02, .02)
    y0 = rng.uniform(-.04, .04)
    yaw0 = 0.0 if start_x is not None else rng.uniform(-.02, .02)
    reset(x0, y0, yaw0, joint_noise=.003)
    policy.reset()
    x_start = float(data.qpos[0])
    reached_at = None
    settled_since = None
    settled_seconds = 0.
    fall = False
    fall_reason = None
    termination_reason = "max_steps"
    holding = False
    hold_tick = None
    hold_from = None
    body_ground = False
    max_tilt = 0.
    min_z = 9.
    wall_touch = False
    max_speed = 0.
    path = []
    trace = []
    proc = None
    renderer = None
    cam = None
    if video_path:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.Popen([
            exe, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "800x600",
            "-r", "25", "-i", "-", "-an", "-vcodec", "libx264",
            "-pix_fmt", "yuv420p", video_path], stdin=subprocess.PIPE)
        renderer = mujoco.Renderer(model, height=600, width=800)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance, cam.azimuth, cam.elevation = 1.0, 135., -12.
    tick = -1
    for tick in range(max_ticks):
        x, y = float(data.qpos[0]), float(data.qpos[1])
        dist = math.hypot(x - TARGET_X, y - TARGET_Y)
        if reached_at is None:
            if x >= stop_x:
                reached_at = tick * .02
                holding = True
                hold_tick = tick
                hold_from = policy.targets.copy()
                cmd = np.zeros(3, np.float32)
            else:
                ey = TARGET_Y - y
                cmd_x = walk_speed(x)
                guide_eff = min(guide_max, GUIDE_RATIO * max(cmd_x, 1e-6))
                vy = float(np.clip(guide_gain * ey, -guide_eff, guide_eff))
                cmd = np.array([cmd_x, vy, 0.], np.float32)
        else:
            cmd = np.zeros(3, np.float32)
        if holding:
            blend = min(1.0, (tick - hold_tick) * .02 / max(hold_seconds, 1e-6))
            target = np.clip(hold_from + blend * (policy.default - hold_from),
                             policy.limits[:, 0], policy.limits[:, 1])
            # Optional DIY finish celebration: a small 10-joint leg bob while
            # both feet remain on the ground, followed by the normal hold.
            if celebrate:
                ht = (tick - hold_tick) * .02
                if .50 <= ht < 1.70:
                    s = math.sin(2 * math.pi * (ht - .50) / .60)
                    off = np.zeros(10)
                    off[2] = off[7] = .07 * s
                    off[3] = off[8] = -.10 * abs(s)
                    off[4] = off[9] = .03 * s
                    target = np.clip(target + off, policy.limits[:, 0], policy.limits[:, 1])
        else:
            obs = policy.build(sensor("gyro"), sensor("accelerometer"), cmd,
                               data.qpos[qaddr], data.qvel[daddr], contacts(),
                               heading=heading())
            raw = policy.infer(obs)
            target = policy.target(raw)
            policy.commit(raw, target)
        data.ctrl[:] = target
        for _ in range(10):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        x, y = float(data.qpos[0]), float(data.qpos[1])
        speed = math.hypot(float(data.qvel[0]), float(data.qvel[1]))
        max_speed = max(max_speed, speed)
        r = data.site_xmat[imu].reshape(3, 3)
        tilt = math.degrees(math.acos(max(-1., min(1., float(r[2, 2])))))
        max_tilt = max(max_tilt, tilt)
        min_z = min(min_z, float(data.qpos[2]))
        if data.qpos[2] < .05:
            fall = True
            fall_reason = "root_z<0.05"
        elif tilt > 75.:
            fall = True
            fall_reason = "tilt>75"
        if wall_hit():
            wall_touch = True
        if body_ground_hit():
            body_ground = True
        if reached_at is not None:
            if dist <= .10 and speed < .02:
                if settled_since is None:
                    settled_since = tick * .02
                settled_seconds = tick * .02 - settled_since
            else:
                settled_since = None
                settled_seconds = 0.
        path.append([x, y])
        trace.append([round(tick * .02, 3), round(x, 5), round(y, 5), round(speed, 5),
                      round(float(cmd[0]), 4), round(float(cmd[1]), 4),
                      round(tilt, 3), int(contacts()[0]), int(contacts()[1]),
                      round(float(data.qpos[2]), 5)])
        if renderer is not None and tick % 2 == 0:
            cam.lookat[:] = data.qpos[:3] + np.array([0., 0., .08])
            renderer.update_scene(data, camera=cam)
            proc.stdin.write(renderer.render().tobytes())
        if fall or wall_touch or body_ground or settled_seconds >= 2.0:
            if fall:
                termination_reason = fall_reason
            elif wall_touch:
                termination_reason = "wall_hit"
            elif body_ground:
                termination_reason = "body_ground_hit"
            else:
                termination_reason = "settled"
            break
    if proc:
        proc.stdin.close()
        proc.wait()
        renderer.close()
    final_dist = math.hypot(float(data.qpos[0]) - TARGET_X, float(data.qpos[1]) - TARGET_Y)
    elapsed = (tick + 1) * .02
    avg_speed = (float(data.qpos[0]) - x_start) / max(elapsed, 1e-6)
    success = (reached_at is not None and final_dist <= .10 and
               settled_seconds >= 2.0 and not fall and not wall_touch and not body_ground)
    return dict(seed=int(seed), x0=float(x0), y0=float(y0), yaw0=float(yaw0),
                success=bool(success), fell=bool(fall), wall_hit=bool(wall_touch),
                body_ground_hit=bool(body_ground), fall_reason=fall_reason,
                termination_reason=termination_reason, hold_used=bool(reached_at is not None),
                reached_at_s=reached_at, final_distance_m=float(final_dist),
                settled_seconds=float(settled_seconds), elapsed_s=float(elapsed),
                avg_speed_m_s=float(avg_speed), max_speed_m_s=float(max_speed),
                max_tilt_deg=float(max_tilt), min_root_z=float(min_z),
                final_xy=[float(data.qpos[0]), float(data.qpos[1])],
                path=path, trace=trace, video=video_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=DEMO + "/models/duck_initial/policy.onnx")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--tag", default="flat5m_duck")
    ap.add_argument("--stop-x", type=float, default=5.02)
    ap.add_argument("--guide-gain", type=float, default=1.0)
    ap.add_argument("--guide-max", type=float, default=.04)
    ap.add_argument("--hold-seconds", type=float, default=.45)
    ap.add_argument("--seed-base", type=int, default=90000)
    ap.add_argument("--terrain", choices=["flat", "slope5", "slope6", "slope8", "slope10"], default="flat")
    ap.add_argument("--celebrate", action="store_true")
    ap.add_argument("--start-x", type=float, default=None)
    ap.add_argument("--max-seconds", type=float, default=140.0)
    ap.add_argument("--backend", choices=["onnx", "rebuilt"], default="onnx")
    ap.add_argument("--rebuilt-params", default=DEMO + "/models/duck_initial/rebuilt_actor.npz")
    args = ap.parse_args()
    init_model(args.terrain)
    if args.backend == "rebuilt":
        policy = RebuiltHeadingPolicy.load(args.rebuilt_params, args.policy.rsplit(".", 1)[0] + ".json")
    else:
        policy = adapter.load(args.policy)
    policy.warmup()
    results = []
    outdir = os.path.join(DEMO, "out")
    os.makedirs(outdir, exist_ok=True)
    for i in range(args.trials):
        vp = os.path.join(outdir, f"{args.tag}_trial{i}.mp4") if args.video else None
        print("TRIAL", i, "video", vp, flush=True)
        r = run_trial(policy, args.seed_base + i, vp, args.stop_x, args.guide_gain,
                      args.guide_max, args.hold_seconds, args.celebrate, args.start_x,
                      int(round(args.max_seconds / .02)))
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("path", "trace")}, indent=2), flush=True)
    summary = dict(policy=args.policy, target=[TARGET_X, TARGET_Y], trials=results,
                   success_count=sum(r["success"] for r in results),
                   trial_count=len(results),
                   terrain=args.terrain, celebrate=args.celebrate,
                   note="continuous 5 m track; physical ramps/walls; no reset; qpos metrics")
    path = os.path.join(outdir, f"{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("WROTE", path)


if __name__ == "__main__":
    main()
