"""Run the enclosed six-grade maze and celebrate at its single exit."""

import argparse
import json
import math
from pathlib import Path

from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from blind_test import BlindPilot
from evaluate import A, BASE, BEAMS, CONTRACT, HERE, INVOCATION_DIR, RebuiltHeadingPolicy
from evaluate import MotionCommand, build_world, mujoco, np, run_case, sha, wrap
from heading_nav import ground_height
from scan_planner import BODY_MARGIN, CELL, NEIGHBORS, RangePlanner
from slope10_finish_celebration import celebrate
from slope_preview import BalancePreview


MAP = HERE / "map_center_exit.json"


class CorridorPlanner(RangePlanner):
    """Build a vehicle-clearance graph from scans, with compiled shortest-path search."""

    def replan(self, position):
        if self.scan_count == 0:
            return False
        shape = self.walls.shape
        self.clearance = CELL * distance_transform_edt(~self.walls)
        blocked = self.clearance < BODY_MARGIN
        cost = 1. + .75 * (~self.free) + .09 / np.maximum(self.clearance, BODY_MARGIN)
        source_xy = self.index(position)
        goal_xy = self.index(self.goal)
        source, goal = np.ravel_multi_index(np.column_stack((source_xy, goal_xy)), shape)
        cells = np.indices(shape).reshape(2, -1).T
        if not hasattr(self, "edges"):
            self.edges = []
            for dx, dy in NEIGHBORS:
                dest = cells + (dx, dy)
                valid = self.valid(dest)
                here = cells[valid]
                there = dest[valid]
                side_x = here + (dx, 0)
                side_y = here + (0, dy)
                indexes = [np.ravel_multi_index(item.T, shape)
                           for item in (here, there, side_x, side_y)]
                self.edges.append((*indexes, math.hypot(dx, dy), dx and dy))
        bad = blocked.ravel()
        weight = cost.ravel()
        links = []
        for here, there, side_x, side_y, length, diagonal in self.edges:
            live = (~bad[here] | (here == source)) & ~bad[there]
            if diagonal:
                live &= ~bad[side_x] & ~bad[side_y]
            links.append((here[live], there[live], length * weight[there[live]]))
        src, dst, values = (np.concatenate(items) for items in zip(*links))
        graph = csr_matrix((values, (src, dst)), shape=(bad.size, bad.size))
        distance, previous = dijkstra(graph, indices=source, return_predecessors=True)
        self.path = np.empty((0, 2))
        if not np.isfinite(distance[goal]):
            return False
        route = [goal]
        while route[-1] != source:
            route.append(int(previous[route[-1]]))
        self.path = self.origin + CELL * cells[route[::-1]]
        return True


class CenterPilot(BlindPilot):
    stop_x = math.inf
    max_seconds = 520.
    start_jitter = (0., 0., .024)
    arrival_command_radius = .046
    use_preview = False
    sequence_driving = False
    descent_speed = .038

    def configure_policy(self, actor):
        self.preview = BalancePreview(actor, self)
        if self.use_preview:
            self.preview.sequence_search = self.sequence_driving
            actor.infer = self.preview

    def __init__(self, goal, profile):
        super().__init__(goal, profile)
        self.sensor_range = 1.65
        self.planner = CorridorPlanner(goal, BEAMS, self.sensor_range, padding=(5.10, 2.6))
        self.grade_direction = None
        self.grade_y = None
        self.approach_grade = 0.

    def ramp_centre_y(self, x, y, direction):
        # Estimate the lane center from the wall returns alongside the slope.
        if self.planner.origin is None or self.planner.walls is None:
            return y
        cells = self.planner.index(np.asarray([x + direction * .42, y]))
        col, _ = cells
        edges = np.flatnonzero(np.any(self.planner.walls[max(0, col - 3):col + 4], axis=0))
        edge_y = self.planner.origin[1] + CELL * edges
        lo = edge_y[edge_y < y - .045]
        hi = edge_y[edge_y > y + .045]
        if not len(lo) or not len(hi) or not .52 <= hi[0] - lo[-1] <= 3.14:
            return y
        return float((lo[-1] + hi[0]) / 2.)

    def terrain_speed(self, pose, desired, t):
        x, y, yaw = pose
        height = float(A.data.qpos[2]) + .24
        if self.grade_direction is None:
            cos = math.cos(desired)
            if abs(cos) > .74:
                direction = math.copysign(1., cos)
                near = ground_height(x + .22 * direction, y, height)
                far = ground_height(x + .49 * direction, y, height)
                if near is not None and far is not None and far - near > .011:
                    self.grade_direction = direction
                    self.grade_y = y
                    if self.sequence_driving:
                        self.grade_y = .65 * y + .35 * self.ramp_centre_y(x, y, direction)
                    self.approach_grade = math.degrees(math.atan2(far - near, .27))
                    self.event("slope_heading_locked", t, pose, yaw_reference_deg=0 if direction > 0 else 180)
        if self.grade_direction is None:
            return super().terrain_speed(pose, desired, t)
        direction = self.grade_direction
        floor = [ground_height(x + direction * dx, y, height) for dx in (-.22, -.08, .12, .49)]
        if all(z is not None for z in floor):
            if max(floor) < .005:
                self.grade_direction = None
                return super().terrain_speed(pose, desired, t)
            self.grade = math.degrees(math.atan2(floor[2] - floor[1], .20))
            self.approach_grade = max(self.approach_grade, abs(self.grade))
            mode = "uphill" if self.grade > 2. else "downhill" if self.grade < -2. else "level"
            if mode != self.last_surface:
                self.event("surface_mode", t, pose, mode=mode, measured_grade_deg=self.grade)
                self.last_surface = mode
        if self.last_surface == "downhill":
            return .046 if self.sequence_driving and self.grade_direction < 0 else self.descent_speed
        if self.sequence_driving and self.grade_direction < 0:
            return .043
        return .055 if self.grade < 7. else .047

    def steer(self, yaw, desired, speed, rate=1., turn_limit=None):
        if self.grade_direction is not None:
            correction = float(np.clip((self.grade_y - A.data.qpos[1]) * .95, -.18, .18))
            desired = math.atan2(correction, self.grade_direction)
            if self.sequence_driving and self.grade_direction < 0:
                self.pivoting = False
                self.heading_reference = self.walk_heading = desired
                error = abs(wrap(desired - yaw))
                reduced = speed * float(np.clip(1.13 - .37 * error, .60, 1.))
                self.forward_command = min(reduced, self.forward_command + .002)
                return MotionCommand(self.forward_command, 0., 0., 1., "p070")
            rate, turn_limit = 1., .28
        return super().steer(yaw, desired, speed, rate, turn_limit)


def geometry(layout):
    model = build_world(layout)
    A.model = model
    A.data = mujoco.MjData(model)
    mujoco.mj_forward(model, A.data)
    measured = []
    for band in layout["terrain"]["grade_bands"]:
        y = sum(band["y_bounds"]) / 2.
        up, roof, down = band["ascent"], band["crest"], band["descent"]
        start, direction = band["start_x"], band["direction"]
        slopes = []
        for offset, length in ((0., up), (up + roof, down)):
            values = [ground_height(start + direction * (offset + length * f), y, .35) for f in (.25, .75)]
            slopes.append(math.degrees(math.atan2(values[1] - values[0], length / 2)))
        if not np.allclose(slopes, [band["degrees"], -band["degrees"]], atol=.10, rtol=0):
            raise ValueError(f"Unexpected physical slope: {band['degrees']}, {slopes}")
        measured.append(dict(degrees=band["degrees"], measured_up_deg=slopes[0], measured_down_deg=slopes[1]))
    return measured


def trial(layout, seed, *, pilot_type=CenterPilot):
    bands = layout["terrain"]["grade_bands"]
    contacts = {str(band["degrees"]): {"up": {}, "down": {}} for band in bands}
    ground = None
    soles = {}
    next_update = 30.
    navigator = None

    def pilot_factory(goal, profile):
        nonlocal navigator
        navigator = pilot_type(goal, profile)
        return navigator

    def observe(frame):
        nonlocal ground, next_update
        if frame["t"] >= next_update:
            print(seed, "t=", round(frame["t"], 1), "xy=",
                  np.round(frame["data"].qpos[:2], 2).tolist(), flush=True)
            next_update += 30.
        if ground is None:
            ground = set(A.ground)
            soles.update(dict(zip(A.feet, ("left", "right"))))
        for c in frame["data"].contact:
            a, b = int(c.geom1), int(c.geom2)
            foot = soles.get(b if a in ground else a if b in ground else -1)
            if foot is None or c.dist > 0:
                continue
            x, y = c.pos[:2]
            for band in bands:
                if not band["y_bounds"][0] + .03 < y < band["y_bounds"][1] - .03:
                    continue
                u = (x - band["start_x"]) * band["direction"]
                up, roof, down = band["ascent"], band["crest"], band["descent"]
                if .08 < u < up - .07:
                    side = "up"
                elif up + roof + .07 < u < up + roof + down - .08:
                    side = "down"
                else:
                    continue
                hits = contacts[str(band["degrees"])][side]
                hits.setdefault(foot, frame["t"])

    row, trace = run_case(layout, seed, "normal", record=True, navigator_type=pilot_factory,
                          measure_gait=True, physical_observer=observe)
    row["balance_preview"] = dict(forecasts=navigator.preview.forecasts,
                                  assisted_ticks=navigator.preview.applied,
                                  candidate_rollouts=navigator.preview.candidate_evaluations,
                                  interventions=navigator.preview.event_log)
    report_speed = getattr(navigator, "speed_report", None)
    if report_speed is not None:
        row["adaptive_speed"] = report_speed()
    row["slope_contacts_s"] = contacts
    goal = tuple(layout["goal_xy"])
    exit_info = layout["exit"]
    dance_frames = []
    dance_tilt = []

    def record(pose, tilt):
        dance_frames.append(pose)
        dance_tilt.append(tilt)

    actor = RebuiltHeadingPolicy.load(str(BASE), str(CONTRACT))
    nav = dict(path=trace["path"][:, 1:3], seconds=row["seconds"], strict_nav_success=row["goal_reached"])
    dance = celebrate(actor, nav, record, amplitude=1.7, goal=goal, finish_line_x=exit_info["x"],
                      direction=exit_info["direction"], exit_y_bounds=exit_info["y_bounds"], sample_dt=.04)
    row["celebration"] = dance
    trace["dance_qpos"] = np.asarray(dance_frames)
    trace["dance_tilt"] = np.asarray(dance_tilt)
    ordered = []
    completed = []
    path = trace["path"]
    for band in bands:
        key = str(band["degrees"])
        for mode in ("up", "down"):
            hit = contacts[key][mode]
            if len(hit) == 2:
                ordered.append(max(hit.values()))
        length = band["ascent"] + band["crest"] + band["descent"]
        lo, hi = band["y_bounds"]
        exited = ((path[:, 1] - band["start_x"]) * band["direction"] >= length
                  ) & (path[:, 2] >= lo) & (path[:, 2] <= hi)
        if exited.any() and all(len(contacts[key][surface]) == 2 for surface in ("up", "down")):
            completed.append(band["degrees"])
    row["slopes_passed"] = completed
    row["fixed_course_pass"] = bool(len(completed) == len(bands) and row["goal_reached"] and dance["pass"]
                                   and len(ordered) == 2 * len(bands)
                                   and ordered == sorted(ordered))
    return row, trace


def main(*, pilot_type=CenterPilot):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[105341, 105342, 105343])
    parser.add_argument("--exit-only", action="store_true",
                        help="Validate the exit and dance from the final flat corridor")
    parser.add_argument("--preview", action="store_true", default=pilot_type is not CenterPilot,
                        help="Test the balance-preview candidate on the steep corridor")
    args = parser.parse_args()
    out = (INVOCATION_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    layout = json.loads(MAP.read_text(encoding="utf-8"))
    axes = layout["enclosure"]
    if layout["start_xy_yaw"][:2] != [(lo + hi) / 2 for lo, hi in (axes["x"], axes["y"])]:
        raise ValueError("Start must be the enclosure centre")
    measured = geometry(layout)
    pilot_type.use_preview = pilot_type.sequence_driving = args.preview
    if args.exit_only:
        x0 = layout["exit"]["x"] + .56
        layout["start_xy_yaw"] = [x0, layout["goal_xy"][1], math.pi]
    print("Physical slopes:", measured, flush=True)
    rows = []
    for seed in args.seeds:
        row, trace = trial(layout, seed, pilot_type=pilot_type)
        row["trace_file"] = f"center_exit_{seed}.npz"
        np.savez_compressed(out / row["trace_file"], **trace)
        rows.append(row)
        print(seed, f"full={row['fixed_course_pass']} arrived={row['goal_reached']}",
              f"fall={row['fell']} collision={row['collision']} t={row['seconds']:.1f}s",
              f"slopes={row['slopes_passed']} dance={row['celebration']['pass']}",
              f"xy={np.round(row['final_xy'], 3)}", flush=True)
    summary = dict(map_id=layout["map_id"], map_file=MAP.name, map_sha256=sha(MAP),
                   actor_sha256=sha(BASE), measured_slopes=measured, physics=mujoco.__version__,
                   scope="exit_and_dance_component" if args.exit_only else "full_maze_from_center",
                   balance_preview_enabled=args.preview,
                   fixed_layout=True, rows=rows, totals=dict(
                       trials=len(rows), full=sum(row["fixed_course_pass"] for row in rows),
                       arrived=sum(row["goal_reached"] for row in rows),
                       danced=sum(row["celebration"]["pass"] for row in rows)))
    if pilot_type is not CenterPilot:
        summary["controller"] = pilot_type.__name__
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
