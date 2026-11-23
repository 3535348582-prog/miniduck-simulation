"""Run the enclosed maze with measured terrain and a gated finish sprint."""

import math

from center_exit_test import A, CenterPilot, main, np, wrap
from heading_nav import ground_height


class AdaptiveCenterPilot(CenterPilot):
    cruise_speed = .115
    sprint_speed = .153
    final_corridor_distance = 1.75
    preview_offsets = (-.23, -.12, 0., .12, .24, .36, .48, .60)

    def __init__(self, goal, profile):
        super().__init__(goal, profile)
        self.flat_since = None
        self.last_hits = None
        self.pace_mode = None
        self.flat_confirmed = False
        self.local_grade = None
        self.forward_grade = None
        self.approaching_grade = False
        self.finish_braking = False
        self.max_cruise_command = 0.
        self.max_sprint_command = 0.

    def command(self, pose, t, hits=None):
        if hits is not None:
            self.last_hits = hits
        return super().command(pose, t, hits)

    def terrain_speed(self, pose, desired, t):
        speed = super().terrain_speed(pose, desired, t)
        x, y, _ = pose
        z = float(A.data.qpos[2]) + .28
        dx, dy = math.cos(desired), math.sin(desired)
        heights = [ground_height(x + offset * dx, y + offset * dy, z)
                   for offset in self.preview_offsets]
        self.approaching_grade = False
        if all(height is not None for height in heights):
            delta = np.diff(self.preview_offsets)
            grades = np.degrees(np.arctan2(np.diff(heights), delta))
            self.local_grade = float(np.degrees(np.arctan2(heights[3] - heights[1], .24)))
            ahead = grades[3:]
            self.forward_grade = float(ahead[np.argmax(abs(ahead))])
            self.approaching_grade = abs(self.local_grade) < 1.7 and abs(self.forward_grade) >= 2.8
            flat = self.grade_direction is None and np.max(np.abs(grades)) < 1.4
        else:
            self.local_grade = self.forward_grade = None
            flat = False
        if flat:
            if self.flat_since is None:
                self.flat_since = t
        else:
            self.flat_since = None
        self.flat_confirmed = self.flat_since is not None and t - self.flat_since >= .24
        return speed

    def flat_drive_ready(self):
        return self.flat_confirmed

    def steer(self, yaw, desired, speed, rate=1., turn_limit=None):
        pos = np.asarray(A.data.qpos[:2])
        t = float(A.data.time)
        pose = (float(pos[0]), float(pos[1]), yaw)
        delta = self.goal - pos
        distance = float(np.linalg.norm(delta))
        goal_yaw = math.atan2(delta[1], delta[0])
        aligned = abs(wrap(desired - yaw)) < .18
        clear = False
        flat_ready = self.flat_drive_ready()
        if self.last_hits is not None and flat_ready and aligned:
            forward = np.array([math.cos(desired), math.sin(desired)])
            lateral = np.array([-forward[1], forward[0]])
            offsets = self.last_hits - pos
            front = offsets @ forward
            sides = abs(offsets @ lateral)
            clear = not np.any((front > 0.) & (front < .52) & (sides < .23))
        mode = "guarded"
        if self.grade_direction is not None:
            if self.last_surface in ("uphill", "downhill"):
                mode = self.last_surface
            else:
                mode = "approach_slope" if self.approaching_grade else "slope_transition"
        elif self.approaching_grade:
            mode = "approach_slope"
        elif flat_ready and distance <= .38:
            self.finish_braking = True
            mode, rate = "finish_brake", 1.
            if clear and abs(wrap(goal_yaw - desired)) < .17:
                speed = min(.116, .046 + .25 * max(0., distance - .10))
        elif clear:
            heading_to_goal = abs(wrap(goal_yaw - desired)) < .17
            if heading_to_goal and not self.finish_braking and .38 < distance < .94:
                mode, speed, rate = "finish_sprint", self.sprint_speed, 1.08
            elif distance >= .50:
                mode = "flat_cruise"
                if heading_to_goal and distance <= self.final_corridor_distance:
                    speed, rate = self.cruise_speed, 1.07
        if mode != self.pace_mode:
            self.pace_mode = mode
            self.event("adaptive_speed_mode", t, pose, mode=mode,
                       command_speed_m_s=speed, goal_distance_m=distance,
                       nearby_grade_deg=self.local_grade, lookahead_grade_deg=self.forward_grade)
        command = super().steer(yaw, desired, speed, rate, turn_limit)
        if mode == "finish_sprint":
            self.max_sprint_command = max(self.max_sprint_command, command.vx)
        elif mode == "flat_cruise":
            self.max_cruise_command = max(self.max_cruise_command, command.vx)
        return command

    def speed_report(self):
        return dict(max_cruise_command_m_s=self.max_cruise_command,
                    max_sprint_command_m_s=self.max_sprint_command,
                    modes=[event for event in self.events if event["event"] == "adaptive_speed_mode"])


if __name__ == "__main__":
    report = main(pilot_type=AdaptiveCenterPilot)
    counts = report["totals"]
    passed = counts["danced"] if report["scope"] == "exit_and_dance_component" else counts["full"]
    if passed != counts["trials"]:
        raise SystemExit(1)
