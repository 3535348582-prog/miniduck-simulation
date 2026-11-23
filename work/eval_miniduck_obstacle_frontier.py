"""Initial obstacle detection and route-choice evaluation for the current Duck actor."""

import argparse
import json
import os
from pathlib import Path

INVOCATION_DIR = Path.cwd()

from slope10_predictive_snapshots_20261123 import (
    A, BASE, CONTRACT, CoordinateGoalAdapter, GoalSpec, MissionConfig, MotionCommand,
    RebuiltHeadingPolicy, load_contract, mission, mujoco, np, original_p070, run_mission, sha,
)
import r4_profile
import r4_terrain


GOAL = (2.1, 0.)
LANES = {"left": .34, "right": -.34}
BOXES = {
    "clear": (),
    "right_open": ((1.04, 0., .10, .105, .13), (1.04, .38, .10, .21, .13)),
    "left_open": ((1.04, 0., .10, .105, .13), (1.04, -.38, .10, .21, .13)),
    "blocked": ((1.04, 0., .10, .59, .13),),
}
EXPECTED = {"clear": "straight", "left_open": "left", "right_open": "right", "blocked": "hold"}
FAN = (1.35, np.linspace(-.78, .78, 41))
OBS_MASK = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)


def fixture(boxes):
    cwd = os.getcwd()
    try:
        os.chdir(r4_terrain.XML_DIR)
        spec = mujoco.MjSpec.from_file(r4_terrain.xml_path(0).name)
        add_fixtures(spec, boxes)
        return spec.compile()
    finally:
        os.chdir(cwd)


def add_fixtures(spec, boxes):
    for i, (x, y, half_x, half_y, half_z) in enumerate(boxes):
        g = spec.worldbody.add_geom()
        g.name = f"assessment_box_{i}"
        g.pos = [x, y, half_z]
        g.size = [half_x, half_y, half_z]
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.contype = g.conaffinity = 1
        g.group = 4
    for name, y in (("wall_left", -.65), ("wall_right", .65)):
        g = spec.worldbody.add_geom()
        g.name = name
        g.pos = [2.5, y, .15]
        g.size = [3.5, .05, .15]
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.contype = g.conaffinity = 1
        g.group = 4


def front_hits(goal):
    xy = A.data.qpos[:2]
    angle = float(np.arctan2(goal[1] - xy[1], goal[0] - xy[0]))
    range_limit, offsets = FAN
    hits = []
    for z in (.07, .14):
        origin = np.array([xy[0], xy[1], z])
        for spread in offsets:
            beam = angle + spread
            direction = np.array([np.cos(beam), np.sin(beam), 0.])
            hit = mujoco.mj_ray(A.model, A.data, origin, direction, OBS_MASK, True,
                                -1, np.empty(1, dtype=np.int32))
            if 0 <= hit <= range_limit:
                hits.append((origin + hit * direction)[:2])
    return np.asarray(hits, float).reshape((-1, 2))


def clear_distance(hits, path):
    if not len(hits):
        return 1.
    nearest = 1.
    for begin, end in zip(path, path[1:]):
        direction = end - begin
        along = np.clip((hits - begin) @ direction / (direction @ direction), 0., 1.)
        nearest = min(nearest, float(np.min(np.linalg.norm(hits - begin - along[:, None] * direction, axis=1))))
    return nearest


def choose(hits, x, y):
    start = np.array([x, y])
    direct = clear_distance(hits, (start, np.array([x + .95, 0.])))
    if direct > .145:
        return "straight", direct
    options = {label: clear_distance(hits, (start, np.array([x + .35, target]),
                                           np.array([x + .95, target])))
               for label, target in LANES.items()}
    side = max(options, key=options.get)
    return (side if options[side] > .145 else "hold"), options[side]


def evaluate(scene, seed, recenter, varied):
    boxes = BOXES[scene]
    if varied and scene != "clear":
        distance = {105240: -.08, 105241: .10, 105242: .16}[seed]
        offset = {105240: .027, 105241: -.034, 105242: .015}[seed]
        if scene == "blocked":
            offset = 0.
        boxes = tuple((x + distance + .035 * i, y + offset, half_x, half_y, half_z)
                      for i, (x, y, half_x, half_y, half_z) in enumerate(boxes))
    contract = load_contract(str(CONTRACT))
    policy = RebuiltHeadingPolicy.load(str(BASE), str(CONTRACT))
    policy.warmup()
    setup = r4_profile.init_custom
    build_base = r4_terrain.model_for_acceptance
    selections = []
    route_choice = "straight"
    passed_obstacle_x = None
    blocked_position = None
    tick = 0
    stable_ticks = 0

    def setup_scene(terrain):
        setup(terrain)
        A.walls.update(A.model.geom(f"assessment_box_{index}").id for index in range(len(boxes)))

    def commands(x, y, cfg):
        nonlocal tick, route_choice, passed_obstacle_x, blocked_position
        tick += 1
        if route_choice == "straight" and passed_obstacle_x is None and x < 1.15 and tick % 5 == 1:
            hits = front_hits(GOAL)
            next_choice, clearance = choose(hits, x, y)
            if next_choice != "straight":
                route_choice = next_choice
                near = hits[(np.abs(hits[:, 1]) < .15)]
                near_x = float(near[:, 0].min()) if len(near) else x + .45
                passed_obstacle_x = near_x + .49
                selections.append(dict(choice=route_choice, clearance_m=clearance,
                                       measured_front_x_m=near_x, time_s=float(A.data.time), x_m=x))
                if route_choice == "hold":
                    blocked_position = x
        if route_choice == "hold":
            return MotionCommand(0., 0., 0., 1., "brake")
        past_obstacle = passed_obstacle_x is not None and x >= passed_obstacle_x
        if recenter and past_obstacle and abs(y) > .045:
            return MotionCommand(.035, float(np.clip(-.90 * y, -.070, .070)), 0., 1., "p070")
        target_y = LANES.get(route_choice, 0.) if not past_obstacle else 0.
        offset = target_y - y
        steering = float(np.clip(.8 * offset, -.044, .044))
        speed = .095 if route_choice == "straight" or x >= (passed_obstacle_x or 999) else .064
        return MotionCommand(speed, steering, 0., 1., "p070")

    r4_terrain.model_for_acceptance = lambda _slope, walls=True: fixture(boxes)
    r4_profile.init_custom = setup_scene
    mission.p070_command = commands
    spec = GoalSpec("explicit_coordinate", arrival_radius_m=.10)
    config = MissionConfig(command_mode="p070", terrain="flat", seed=seed, goal_xy=GOAL,
                           stop_x=GOAL[0], max_seconds=18. if scene=="blocked" else 55.,
                           contract=contract, trace=False, render_every=1)

    def observe(event):
        nonlocal stable_ticks
        stable = (route_choice == "hold" and np.linalg.norm(A.data.qvel[:2]) < .02
                  and np.linalg.norm(A.data.qvel[3:6]) < .20 and bool(np.all(A.contacts())))
        stable_ticks = stable_ticks + 1 if stable else 0

    try:
        result = run_mission(policy, CoordinateGoalAdapter(spec, GOAL), spec, config, observer=observe)
    finally:
        mission.p070_command = original_p070
        r4_profile.init_custom = setup
        r4_terrain.model_for_acceptance = build_base
    collision = bool(result["wall_hit"] or result["body_ground_hit"])
    farthest = max(x for x, _ in result["path"])
    follows_choice = EXPECTED[scene] == route_choice
    stopped = (blocked_position is not None and farthest < .86 and stable_ticks >= 100
               and not collision and not result["fell"])
    return dict(scene=scene, seed=seed, expected=EXPECTED[scene], choice=route_choice,
                events=selections, decision_correct=follows_choice, collision=collision, fell=result["fell"],
                physically_passed=bool(result["strict_nav_success"] and farthest >= GOAL[0]),
                crossed_finish=bool(farthest >= GOAL[0]), final_xy=result["path"][-1],
                safe_hold=stopped, final_error_m=result["final_distance_m"],
                reached_x_m=farthest, seconds=result["seconds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--recenter", action="store_true")
    parser.add_argument("--varied", action="store_true")
    args = parser.parse_args()
    args.out = (INVOCATION_DIR / args.out).resolve()
    assert mujoco.__version__ == "3.6.0"
    seeds = (105240, 105241, 105242) if args.varied else (105235, 105236, 105237)
    rows = [evaluate(scene, seed, args.recenter, args.varied) for scene in BOXES for seed in seeds]
    result = dict(scope="Initial native simulated range and passage checks on a flat 2.1 m course",
                  actor_sha256=sha(BASE), recenter_after_obstacle=args.recenter,
                  varied_box_locations=args.varied, range_limit_m=FAN[0], rows=rows)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    for scene in BOXES:
        cases = [r for r in rows if r["scene"] == scene]
        print(scene, "decisions", sum(r["decision_correct"] for r in cases),
              "crossed", sum(r["physically_passed"] for r in cases),
              "safe stop", sum(r["safe_hold"] for r in cases), flush=True)
