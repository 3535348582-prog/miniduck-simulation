"""Evaluate physical navigation on the fixed one-goal fork in native MuJoCo."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
INVOCATION_DIR = Path.cwd()
os.environ.setdefault("MINIDUCK_ROOT", str(WORK / "miniduck_native_gate_20261123"))
os.environ.setdefault("MINIDUCK_TRAINING", str(WORK.parent / "miniducktraining_classroom"))
os.environ.setdefault("MINIDUCK_ASSET_ROOT", os.environ["MINIDUCK_TRAINING"])
os.environ.setdefault("MINIDUCK_XML_DIR", str(Path(os.environ["MINIDUCK_TRAINING"])
                      / "training/adapted/playground/open_duck_mini_v2/xmls"))
sys.path[:0] = [str(WORK), str(WORK / "slope10_final_candidate_v1_20261123")]

from eval_miniduck_obstacle_frontier import (
    A, BASE, CONTRACT, CoordinateGoalAdapter, GoalSpec, MissionConfig, MotionCommand,
    OBS_MASK, RebuiltHeadingPolicy, clear_distance, load_contract, mission, mujoco,
    r4_profile, r4_terrain, run_mission, sha,
)

MAP = HERE / "map.json"
RANGE_M = .82
BEAMS = np.linspace(-math.pi, math.pi, 120, endpoint=False)
ROBOT_MARGIN_M = .16
LANE_OFFSET_M = .38
MAX_SECONDS = 125.
PHASES = ("approach", "align", "explore", "turn_back", "retreat", "turn_forward", "finish",
          "settle", "hold")


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def wall_names(layout):
    names = [item["name"] for item in layout["walls"]]
    return names + [f"{side}_{i}" for side, points in layout.get("wall_paths", {}).items()
                    for i in range(len(points) - 1)]


def add_rough_ground(spec, config):
    x0, x1 = config["x_bounds"]
    y0, y1 = config["y_bounds"]
    bumps = config.get("speed_bumps", ())
    grades = config.get("grade_bands", ())
    rounded = config.get("rounded_hills", ())
    xs = np.linspace(x0, x1, max(261, math.ceil((x1 - x0) / .020))
                     if grades or rounded else max(261, math.ceil((x1 - x0) / .022)) if bumps else 261)
    ys = np.linspace(y0, y1, max(61, math.ceil((y1 - y0) / .024)) if grades or rounded else 61)
    centers = np.asarray(config["x_height"], dtype=float)
    start, end = config["roughness_x_range"]
    taper = np.clip((xs - start) / .28, 0., 1.) * np.clip((end - xs) / .28, 0., 1.)
    wavelength_x, wavelength_y = config["roughness_wavelength_m"]
    wave = .57 * np.sin(2. * np.pi * (xs[None, :] / wavelength_x + ys[:, None] / wavelength_y))
    wave += .43 * np.sin(2. * np.pi * (xs[None, :] / (1.38 * wavelength_x)
                                      - ys[:, None] / (1.18 * wavelength_y)))
    relief = config["roughness_amplitude_m"] * taper[None, :] * (1. + wave)
    heights = np.interp(xs, centers[:, 0], centers[:, 1])[None, :] + relief
    for bump in bumps:
        low, high = bump["x_bounds"]
        fraction = np.clip((xs - low) / (high - low), 0., 1.)
        heights += bump["height"] * np.sin(np.pi * fraction)[None, :] ** 2
    for band in grades:
        along = band["direction"] * (xs - band["start_x"])
        up, roof, down = band["ascent"], band["crest"], band["descent"]
        top = up * math.tan(math.radians(band["degrees"]))
        ridge = np.interp(along, [0., up, up + roof, up + roof + down],
                          [0., top, top, 0.], left=0., right=0.)
        lo, hi = band["y_bounds"]
        cross = (ys >= lo) & (ys <= hi)
        heights += cross[:, None] * ridge[None, :]
    for hill in rounded:
        along = hill["direction"] * (xs - hill["start_x"])
        length = hill["length"]
        rise = length * math.tan(math.radians(hill["degrees"])) / math.pi
        ridge = rise * np.sin(np.pi * np.clip(along / length, 0., 1.)) ** 2
        lo, hi = hill["y_bounds"]
        heights += ((ys >= lo) & (ys <= hi))[:, None] * ridge[None, :]
    spec.add_hfield(name="rough_ground_asset", size=[(x1 - x0) / 2, (y1 - y0) / 2,
                    float(heights.max()), .025], nrow=len(ys), ncol=len(xs),
                    userdata=heights.ravel())
    floor = spec.geom("floor")
    floor.pos[2] -= .035
    floor.contype = floor.conaffinity = 0
    spec.worldbody.add_geom(name="rough_ground", type=mujoco.mjtGeom.mjGEOM_HFIELD,
                           hfieldname="rough_ground_asset", pos=[(x1 + x0) / 2, (y1 + y0) / 2, 0.],
                           group=3, contype=1, conaffinity=1, friction=[.78, .02, .01],
                           rgba=[.47, .55, .47, 1.])


def add_planar_ramps(spec, bands):
    spec.geom("floor").group = 3
    for i, band in enumerate(bands):
        lo, hi = band["y_bounds"]
        direction = band["direction"]
        x = band["start_x"]
        up, crest, down = band["ascent"], band["crest"], band["descent"]
        angle = math.radians(band["degrees"])
        apron = band.get("toe_apron", 0.)
        fraction = .40
        shallow = math.atan(fraction * math.tan(angle))
        low_z = apron * math.tan(shallow)
        top = low_z + (up - apron) * math.tan(angle)
        surfaces = [
            ("up", (up + apron) / 2, up - apron, (low_z + top) / 2, direction * angle),
            ("crest", up + crest / 2, crest, top, 0.),
            ("down", up + crest + (down - apron) / 2, down - apron, (low_z + top) / 2, -direction * angle),
        ]
        if apron:
            surfaces += [("entry_apron", apron / 2, apron, low_z / 2, direction * shallow),
                         ("exit_apron", up + crest + down - apron / 2, apron,
                          low_z / 2, -direction * shallow)]
        for part, dx, length, z, slope in surfaces:
            thickness = .020
            spec.worldbody.add_geom(
                name=f"grade_{i}_{part}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[length / (2 * math.cos(slope)) + .010, (hi - lo) / 2, thickness],
                pos=[x + direction * dx + thickness * math.sin(slope),
                     (lo + hi) / 2, z - thickness * math.cos(slope)],
                quat=[math.cos(slope / 2), 0., -math.sin(slope / 2), 0.],
                group=3, contype=1, conaffinity=1, friction=[.80, .02, .01],
                rgba=[.47, .58, .51, 1.])


def build_world(layout, *, decorate=None):
    cwd = os.getcwd()
    try:
        os.chdir(r4_terrain.XML_DIR)
        spec = mujoco.MjSpec.from_file(r4_terrain.xml_path(0).name)
        height = layout["wall_height"]
        if "terrain" in layout:
            terrain = layout["terrain"]
            if terrain.get("planar_ramps"):
                add_planar_ramps(spec, terrain["grade_bands"])
            else:
                add_rough_ground(spec, terrain)
        for item in layout["walls"]:
            x0, x1 = item["x_bounds"]
            y0, y1 = item["y_bounds"]
            geom = spec.worldbody.add_geom()
            geom.name = item["name"]
            geom.pos = [(x0 + x1) / 2, (y0 + y1) / 2, height / 2]
            geom.size = [(x1 - x0) / 2, (y1 - y0) / 2, height / 2]
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.contype = geom.conaffinity = 1
            geom.group = 4
            geom.rgba = item.get("rgba", [.18, .23, .32, 1.])
        for side, points in layout.get("wall_paths", {}).items():
            for i, (start, end) in enumerate(zip(points, points[1:])):
                dx, dy = np.subtract(end, start)
                half_yaw = math.atan2(dy, dx) / 2
                g = spec.worldbody.add_geom(name=f"{side}_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                           size=[math.hypot(dx, dy) / 2 + .018, .03, height / 2],
                                           pos=[(start[0] + end[0]) / 2, (start[1] + end[1]) / 2, height / 2],
                                           quat=[math.cos(half_yaw), 0., 0., math.sin(half_yaw)],
                                           contype=1, conaffinity=1, group=4,
                                           rgba=[.18, .23, .32, 1.])
        gx, gy = layout["goal_xy"]
        marker = spec.worldbody.add_site(name="single_goal")
        marker.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        marker.pos = [gx, gy, .002]
        marker.size = [layout["goal_radius"], .002, .002]
        marker.rgba = [.08, .9, .46, .65]
        if decorate is not None:
            decorate(spec, layout)
        return spec.compile()
    finally:
        os.chdir(cwd)


def observe_ranges(pose, reach=RANGE_M, relative_ground=False):
    x, y, yaw = pose
    z = max(.085, float(A.data.qpos[2]) - .036) if relative_ground else .10
    origin = np.array([x, y, z])
    hits = []
    geom = np.empty(1, dtype=np.int32)
    for beam in BEAMS + yaw:
        direction = np.array([math.cos(beam), math.sin(beam), 0.])
        distance = mujoco.mj_ray(A.model, A.data, origin, direction, OBS_MASK, True, -1, geom)
        if 0. <= distance <= reach:
            hits.append((origin + distance * direction)[:2])
    return np.asarray(hits, dtype=float).reshape((-1, 2))


def move(vx, vy, desired_yaw, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    local_x = c * vx + s * vy
    local_y = -s * vx + c * vy
    turn = float(np.clip(1.5 * wrap(desired_yaw - yaw), -.32, .32))
    return MotionCommand(local_x, local_y, turn, 1., "p070")


def brake():
    return MotionCommand(0., 0., 0., 1., "brake")


class ForkNavigator:
    """Track the junction and rejected branches using pose and local range hits."""

    def __init__(self, goal, preferred):
        self.goal = np.asarray(goal)
        self.preferred = preferred
        self.phase = "approach"
        self.lanes = {"left": self.goal[1] + LANE_OFFSET_M, "right": self.goal[1] - LANE_OFFSET_M}
        self.junction = None
        self.branch = None
        self.blocked = set()
        self.beside_divider = False
        self.events = []
        self.hits = np.empty((0, 2))
        self.turn_goal = 0.
        self.blocked_pose = None

    def event(self, name, t, pose, **extra):
        self.events.append(dict(event=name, time_s=round(t, 3), pose=list(pose), **extra))

    def choose(self, t, pose):
        x, y, _ = pose
        options = {}
        for side, lane in self.lanes.items():
            if side not in self.blocked:
                path = (np.array([x, y]), np.array([x, lane]), np.array([x + .60, lane]))
                options[side] = clear_distance(self.hits, path)
        open_sides = [side for side, gap in options.items() if gap > ROBOT_MARGIN_M]
        if not open_sides:
            self.phase = "hold"
            self.event("no_open_branch", t, pose, clearances_m=options, rejected=sorted(self.blocked))
            return
        best = max(open_sides, key=options.get)
        self.branch = (self.preferred if self.preferred in open_sides
                       and options[best] - options[self.preferred] <= .05 else best)
        self.phase = "align"
        self.beside_divider = False
        self.event("choose_branch", t, pose, branch=self.branch,
                   clearances_m=options, rejected=sorted(self.blocked))

    def command(self, pose, t, hits=None):
        x, y, yaw = pose
        scan = hits is not None
        if scan:
            self.hits = hits
        gy = self.goal[1]
        center_v = float(np.clip(.85 * (gy - y), -.06, .06))

        if self.phase == "approach":
            frontal = self.hits[(self.hits[:, 0] > x) & (abs(self.hits[:, 1] - gy) < .125)]
            if scan and len(frontal) >= 3 and min(frontal[:, 0]) - x < .63:
                self.junction = x
                self.choose(t, pose)
            else:
                return move(.076, center_v, 0., yaw)

        if self.phase == "hold" or self.phase == "settle":
            return brake()
        lane = self.lanes[self.branch]
        cross_v = float(np.clip(.95 * (lane - y), -.06, .06))

        if self.phase == "align":
            if abs(lane - y) <= .030:
                self.phase = "explore"
                self.event("enter_branch", t, pose, branch=self.branch)
            else:
                x_correction = float(np.clip(1.0 * (self.junction - x), -.040, .040))
                return move(x_correction, cross_v, 0., yaw)

        if self.phase == "explore":
            if scan:
                hits_x = self.hits[:, 0] - x
                forward = self.hits[(hits_x > 0) & (hits_x < .36)
                                    & (abs(self.hits[:, 1] - lane) < .118)]
                lateral = self.hits[(self.hits[:, 1] - gy) * np.sign(lane - gy) >= -.04]
                lateral = lateral[(lateral[:, 1] - y) * np.sign(lane - gy) < -.10]
                beside = bool(np.any(abs(lateral[:, 0] - x) <= .12))
                self.beside_divider = self.beside_divider or beside
                if len(forward) >= 4:
                    self.blocked.add(self.branch)
                    self.blocked_pose = pose
                    self.turn_goal = math.pi
                    self.phase = "turn_back"
                    self.event("dead_end_detected", t, pose, branch=self.branch,
                               obstacle_range_m=float(min(forward[:, 0]) - x),
                               hits_in_corridor=len(forward))
                elif self.beside_divider and not np.any(abs(lateral[:, 0] - x) <= .21):
                    self.phase = "finish"
                    self.event("clear_of_divider", t, pose, branch=self.branch)
            if self.phase == "explore":
                return move(.068, cross_v, 0., yaw)

        if self.phase in ("turn_back", "turn_forward"):
            error = wrap(self.turn_goal - yaw)
            if abs(error) > .105:
                turn = math.copysign(float(np.clip(1.4 * abs(error), .125, .32)), error)
                return MotionCommand(0., 0., turn, 1., "turn")
            if self.phase == "turn_back":
                self.phase = "retreat"
                self.event("turned_around", t, pose, branch=self.branch,
                           heading_change_deg=abs(math.degrees(wrap(yaw - self.blocked_pose[2]))))
            else:
                if scan:
                    self.choose(t, pose)
                return brake()

        if self.phase == "retreat":
            if x <= self.junction + .030:
                self.phase = "turn_forward"
                self.turn_goal = 0.
                self.event("returned_to_junction", t, pose, branch=self.branch,
                           retreat_m=self.blocked_pose[0] - x)
                return brake()
            return move(-.068, cross_v, math.pi, yaw)

        dx = self.goal[0] - x
        if self.phase == "finish" and abs(dx) <= .057 and abs(gy - y) < .055:
            self.phase = "settle"
            self.event("entered_goal", t, pose)
            return brake()
        if self.phase == "finish":
            vx = float(np.clip(.7 * dx, -.035, .086))
            if abs(gy - y) > .17:
                vx = min(vx, .034)
            return move(vx, center_v, 0., yaw)
        return brake()


def run_case(layout, seed, preferred, record=False, navigator_type=ForkNavigator, measure_gait=False,
             physical_observer=None):
    goal = tuple(layout["goal_xy"])
    rng = np.random.default_rng(seed)
    nav = navigator_type(goal, preferred)
    start = np.asarray(layout["start_xy_yaw"]) + rng.uniform(-1, 1, 3) * getattr(
        nav, "start_jitter", [.016, .032, .028])
    samples, poses, phase_ids, ray_hits, scans_at = [], [], [], [], []
    gait = []
    policy = RebuiltHeadingPolicy.load(str(BASE), str(CONTRACT))
    policy.warmup()
    if measure_gait:
        base_commit = policy.commit
        base_build = policy.build

        def commit(action, target):
            before = policy.phase
            base_commit(action, target)
            policy.phase = (before + getattr(nav, "phase_rate", 1.)) % policy.contract["phase_period_steps"]

        policy.commit = commit

        def build(gyro, accel, command, q, dq, contacts, heading=None):
            reference = getattr(nav, "heading_reference", None)
            obs = base_build(gyro, accel, command, q, dq, contacts, heading=heading)
            if reference is not None and command[0] > .01 and command[2] == 0.:
                error = float(np.clip(wrap(float(heading) - reference), -.40, .40))
                obs[-2:] = [math.sin(error), math.cos(error)]
            return obs

        policy.build = build
    configure_policy = getattr(nav, "configure_policy", None)
    if configure_policy is not None:
        configure_policy(policy)

    contract = load_contract(CONTRACT)
    spec = GoalSpec("explicit_coordinate", arrival_radius_m=layout["goal_radius"])
    config = MissionConfig(command_mode="p070", terrain="flat", seed=None,
                           start_xyz=tuple(start), goal_xy=goal,
                           stop_x=getattr(nav, "stop_x", goal[0] + .16),
                           max_seconds=getattr(nav, "max_seconds", MAX_SECONDS),
                           contract=contract, trace=False,
                           render_every=1 if measure_gait else 2)
    setup = r4_profile.init_custom
    model = build_world(layout)
    ticks = 0

    def setup_maze(terrain):
        setup(terrain)
        A.walls.update(A.model.geom(name).id for name in wall_names(layout))
        if "terrain" in layout:
            if layout["terrain"].get("planar_ramps"):
                A.ground.extend(i for i in range(A.model.ngeom)
                                if A.model.geom(i).name.startswith("grade_"))
            else:
                A.ground.append(A.model.geom("rough_ground").id)

    def command(x, y, _config):
        nonlocal ticks
        pose = (float(x), float(y), float(A.heading()))
        hits = (observe_ranges(pose, getattr(nav, "sensor_range", RANGE_M), "terrain" in layout)
                if ticks % 5 == 0 else None)
        if record and hits is not None:
            ray_hits.extend(hits.tolist())
            scans_at.append([ticks, len(ray_hits)])
        ticks += 1
        return nav.command(pose, float(A.data.time), hits)

    def observe(frame):
        if physical_observer is not None:
            physical_observer(frame)
        if measure_gait:
            cmd = frame["cmd"]
            gait.append((frame["t"], PHASES.index(nav.phase), cmd.vx, cmd.vy, cmd.yaw_rate,
                         getattr(nav, "phase_rate", 1.), *A.contacts()))
        if frame["tick"] % 2:
            return
        if record:
            poses.append(frame["data"].qpos.copy())
            phase_ids.append(PHASES.index(nav.phase))
        samples.append((frame["t"], frame["x"], frame["y"], float(A.heading())))

    with (patch.object(r4_terrain, "model_for_acceptance", return_value=model),
          patch.object(r4_profile, "init_custom", setup_maze),
          patch.object(mission, "p070_command", command)):
        result = run_mission(policy, CoordinateGoalAdapter(spec, goal), spec, config, observer=observe)

    branches = [e["branch"] for e in nav.events if e["event"] == "choose_branch"]
    turn = next((e["heading_change_deg"] for e in nav.events if e["event"] == "turned_around"), 0.)
    retreat = next((e["retreat_m"] for e in nav.events if e["event"] == "returned_to_junction"), 0.)
    recovered = bool(len(branches) == 2 and branches[0] != branches[1] and branches[0] in nav.blocked
                     and turn >= 160. and retreat >= .40)
    collision = bool(result["wall_hit"] or result["body_ground_hit"])
    reached = result["strict_nav_success"] and nav.phase == "settle"
    row = dict(seed=seed, preferred_when_equal=preferred, start_xyz=list(start), chosen_branches=branches,
               blocked_branches=sorted(nav.blocked), dead_end_recovered=recovered, turned_deg=turn,
               retreat_m=retreat, final_xy=result["path"][-1], final_error_m=result["final_distance_m"],
               seconds=result["seconds"], max_tilt_deg=result["max_tilt_deg"], collision=collision,
               fell=result["fell"], fall_reason=result["fall_reason"], phase=nav.phase,
               goal_reached=bool(reached), done=bool(reached and (recovered if preferred == "left" else True)),
               events=nav.events)
    trace = dict(path=np.asarray(samples))
    if measure_gait:
        trace["gait"] = np.asarray(gait)
    if record:
        trace.update(qpos=np.asarray(poses), phase=np.asarray(phase_ids),
                     ray_hits=np.asarray(ray_hits).reshape(-1, 2), scans_at=np.asarray(scans_at))
    return row, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[105250, 105251, 105252])
    parser.add_argument("--preferences", nargs="+", choices=["left", "right"], default=["left", "right"])
    args = parser.parse_args()
    out = (INVOCATION_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with MAP.open(encoding="utf-8") as stream:
        layout = json.load(stream)
    rows = []
    for preferred in args.preferences:
        for index, seed in enumerate(args.seeds):
            row, trace = run_case(layout, seed, preferred, record=index == 0)
            name = f"{preferred}_{seed}.npz"
            np.savez_compressed(out / name, **trace)
            row["trace_file"] = name
            rows.append(row)
            print(preferred, seed, row["chosen_branches"], row["phase"], row["goal_reached"],
                  f"distance={row['final_error_m']:.3f}m", f"time={row['seconds']:.2f}s",
                  f"collision={row['collision']}", f"fall={row['fell']}", flush=True)

    report = dict(map_id=layout["map_id"], map_sha256=sha(MAP), evaluator_sha256=sha(Path(__file__)),
                  actor_sha256=sha(BASE), contract_sha256=sha(CONTRACT), physics=mujoco.__version__,
                  range_limit_m=RANGE_M, range_rays=len(BEAMS), sensors="simulated range rays and exact pose",
                  arrival_radius_m=layout["goal_radius"], settle_s=2.,
                  totals=dict(trials=len(rows), passed=sum(row["done"] for row in rows),
                              reached=sum(row["goal_reached"] for row in rows),
                              collision=sum(row["collision"] for row in rows), falls=sum(row["fell"] for row in rows),
                              recovered=sum(row["dead_end_recovered"] for row in rows)),
                  rows=rows)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["totals"]), flush=True)


if __name__ == "__main__":
    main()
