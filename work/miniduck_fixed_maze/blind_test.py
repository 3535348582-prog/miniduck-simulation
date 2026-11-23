"""Paired, map-blind native trials with two physical switchable dead ends."""

import argparse
import json
import math
from pathlib import Path

from evaluate import A, BASE, BEAMS, CONTRACT, HERE, INVOCATION_DIR, np, run_case, sha, wrap
from heading_nav import HeadingNavigator, brake, ground_height
from scan_planner import RangePlanner


MAP = HERE / "map_multifork.json"
GATES = {"A": ("left", "right"), "B": ("right", "left"),
         "C": ("left", "left"), "D": ("right", "right")}
SENSORS = {"normal": 1.40, "short": .64, "off": 0.}


def physical_layout(fixture, case):
    result = dict(fixture)
    caps = []
    for index, (gate, blocked) in enumerate(zip(fixture["switches"], GATES[case]), start=1):
        y = gate["inner_y"]
        caps.append(dict(name=f"cap_{index}", x_bounds=gate["x_bounds"],
                         y_bounds=[y, .71] if blocked == "left" else [-.71, -y]))
    result["walls"] = fixture["walls"] + caps
    return result


class BlindPilot(HeadingNavigator):
    """The planner never receives wall geometry or gate labels."""

    max_seconds = 275.
    arrival_command_radius = .065

    def __init__(self, goal, sensor_profile):
        super().__init__(goal, sensor_profile)
        self.sensor_range = SENSORS[sensor_profile]
        self.planner = RangePlanner(goal, BEAMS, self.sensor_range)
        self.next_plan = 0.
        self.previous_direction = None
        self.grade = 0.
        self.slope_recent = 0.
        self.plan_count = 0
        if sensor_profile == "off":
            self.max_seconds = 4.

    def terrain_speed(self, pose, desired, t):
        x, y, _ = pose
        z = float(A.data.qpos[2]) + .28
        dx, dy = .18 * math.cos(desired), .18 * math.sin(desired)
        foot = ground_height(x - dx, y - dy, z)
        look = ground_height(x + dx, y + dy, z)
        if foot is not None and look is not None:
            sample = math.degrees(math.atan2(look - foot, .36))
            self.grade += .25 * (sample - self.grade)
        if abs(self.grade) > 1.9:
            self.slope_recent = t + 2.0
        if self.grade < -2.:
            speed, mode = .039, "downhill"
        elif self.grade > 2.:
            speed, mode = .055, "uphill"
        elif t < self.slope_recent:
            speed, mode = .053, "level"
        else:
            speed, mode = .100, "level"
        if mode != self.last_surface:
            self.event("surface_mode", t, pose, mode=mode, measured_grade_deg=self.grade)
            self.last_surface = mode
        return speed

    def command(self, pose, t, hits=None):
        position = np.asarray(pose[:2])
        yaw = pose[2]
        self.phase_rate = 1.
        self.heading_reference = None
        if self.phase in ("hold", "settle"):
            return brake()
        if self.sensor_range == 0.:
            self.phase = "hold"
            self.event("range_unavailable", t, pose)
            return brake()
        if hits is not None:
            self.planner.update(position, yaw, hits)
        if math.dist(position, self.goal) <= self.arrival_command_radius:
            self.phase = "settle"
            self.event("entered_goal", t, pose)
            return brake()
        if t >= self.next_plan:
            self.planner.replan(position)
            self.plan_count += 1
            self.next_plan = t + .34
        target = self.planner.lookahead(position, .38)
        if target is None:
            self.forward_command = 0.
            return brake()
        dx, dy = target - position
        angle = math.atan2(dy, dx)
        if self.previous_direction is None or abs(wrap(angle - self.previous_direction)) > math.radians(110.):
            self.event("planned_heading_changed", t, pose, desired_deg=math.degrees(angle),
                       observed_cells=int(np.count_nonzero(self.planner.walls)))
        self.previous_direction = angle
        speed = self.terrain_speed(pose, angle, t) if hits is not None else self.reference_speed
        self.reference_speed = speed
        if math.dist(position, self.goal) < .50:
            speed = min(speed, .060)
        self.phase = "explore"
        return self.steer(yaw, angle, speed, 1.07 if speed >= .095 else 1.)

    reference_speed = .060


def assess(fixture, case, sensor, seed):
    nav = None

    def factory(goal, profile):
        nonlocal nav
        nav = BlindPilot(goal, profile)
        return nav

    row, trace = run_case(physical_layout(fixture, case), seed, sensor, record=True,
                          navigator_type=factory, measure_gait=True)
    path = trace["path"]
    decisions = []
    for x in fixture["decision_planes_x"]:
        crossed = np.flatnonzero(path[:, 1] >= x)
        if crossed.size:
            t, _, y, _ = path[crossed[0]]
            decisions.append(dict(plane_x=x, time_s=t, y_m=y, side="left" if y > 0. else "right"))
    expected = ["right" if blocked == "left" else "left" for blocked in GATES[case]]
    correct = len(decisions) == len(expected) and [d["side"] for d in decisions] == expected
    moved = np.linalg.norm(path[:, 1:3] - np.asarray(row["start_xyz"][:2]), axis=1)
    retreat = np.maximum.accumulate(path[:, 1]) - path[:, 1]
    row.update(layout=case, sensor=sensor, expected_sides=expected, measured_crossings=decisions,
               both_choices_correct=correct, max_reverse_distance_m=float(retreat.max()),
               max_displacement_m=float(moved.max()), scan_count=nav.planner.scan_count,
               plans=nav.plan_count, obstacle_cells=int(np.count_nonzero(nav.planner.walls)),
               safe_sensor_off=bool(sensor == "off" and moved.max() <= .12 and
                                    not row["collision"] and not row["fell"] and row["phase"] == "hold"))
    trace["occupancy"] = nav.planner.walls if nav.planner.walls is not None else np.zeros((0, 0), dtype=bool)
    trace["grid_origin"] = nav.planner.origin if nav.planner.origin is not None else np.zeros(2)
    return row, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[105311, 105312, 105313])
    parser.add_argument("--layouts", nargs="+", choices=GATES, default=list(GATES))
    parser.add_argument("--sensors", nargs="+", choices=SENSORS, default=["normal", "short", "off"])
    args = parser.parse_args()
    output = (INVOCATION_DIR / args.out).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads(MAP.read_text(encoding="utf-8"))
    evidence = dict(map_id=fixture["map_id"], map_sha256=sha(MAP), actor_sha256=sha(BASE),
                    contract_sha256=sha(CONTRACT), sensor_ranges_m=SENSORS,
                    controller_inputs=["simulated pose", "goal coordinate", "range hits", "terrain range"],
                    source_hashes={n: sha(HERE / n) for n in
                                   ("blind_test.py", "scan_planner.py", "heading_nav.py", "evaluate.py")},
                    rows=[])
    for case in args.layouts:
        for sensor in args.sensors:
            for seed in args.seeds:
                row, trace = assess(fixture, case, sensor, seed)
                row["trace_file"] = f"{case}_{sensor}_{seed}.npz"
                np.savez_compressed(output / row["trace_file"], **trace)
                evidence["rows"].append(row)
                print(case, sensor, seed, [d["side"] for d in row["measured_crossings"]],
                      f"goal={row['goal_reached']} fall={row['fell']} collision={row['collision']}",
                      f"backtrack={row['max_reverse_distance_m']:.2f}m "
                      f"time={row['seconds']:.2f}s xy={np.round(row['final_xy'], 2).tolist()}",
                      flush=True)
    (output / "report.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
