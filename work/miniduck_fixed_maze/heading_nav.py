"""Forward-only fork navigation with heading control and terrain-aware cruise."""

import math

import numpy as np

from evaluate import A, ForkNavigator, MotionCommand, brake, mujoco, wrap


def ground_height(x, y, z):
    depth = mujoco.mj_ray(A.model, A.data, np.array([x, y, z]), np.array([0., 0., -1.]),
                          np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8), True, -1,
                          np.empty(1, dtype=np.int32))
    return z - depth if depth >= 0. else None


class HeadingNavigator(ForkNavigator):
    cruise = .120
    cruise_rate = 1.15
    branch_speed = .100
    branch_rate = 1.10
    lane_offset = .31
    turn_rate = .62
    rough_map = True
    max_seconds = 178.

    def __init__(self, goal, preferred):
        super().__init__(goal, preferred)
        self.lanes = {"left": self.lane_offset, "right": -self.lane_offset}
        self.phase_rate = 1.
        self.entry_x = None
        self.grade = 0.
        self.along_grade = 0.
        self.centerline_y = self.lanes["right"]
        self.last_surface = None
        self.forward_command = 0.
        self.heading_reference = None
        self.walk_heading = None
        self.pivoting = False
        self.sensor_range = 1.36 if preferred == "auto" else .82

    def choose(self, t, pose):
        if self.rough_map and self.preferred == "auto":
            x = pose[0]
            for side, lane in self.lanes.items():
                side_hits = self.hits[(self.hits[:, 0] > x + .65)
                                      & (abs(self.hits[:, 1] - lane) < .175)]
                for distance_x in side_hits[:, 0]:
                    cluster = side_hits[abs(side_hits[:, 0] - distance_x) < .022]
                    if len(cluster) >= 3 and np.ptp(cluster[:, 1]) >= .115:
                        if side not in self.blocked:
                            self.blocked.add(side)
                            self.event("branch_rejected_from_scan", t, pose, branch=side,
                                       obstacle_range_m=float(distance_x - x),
                                       observed_width_m=float(np.ptp(cluster[:, 1])))
                        break
        super().choose(t, pose)
        if self.phase == "align":
            self.entry_x = max(pose[0], self.junction) + .22

    def steer(self, yaw, desired, speed, rate=1., turn_limit=None):
        error = wrap(desired - yaw)
        limit = self.turn_rate if turn_limit is None else turn_limit
        self.pivoting = abs(error) > (.29 if self.pivoting else .60)
        if self.pivoting:
            self.forward_command = 0.
            self.walk_heading = None
            turn = float(np.clip(1.9 * error, -limit, limit))
            return MotionCommand(0., 0., turn, 1., "turn")
        if self.walk_heading is None:
            self.walk_heading = yaw
        change = float(np.clip(wrap(desired - self.walk_heading), -.012, .012))
        self.walk_heading = wrap(self.walk_heading + change)
        self.heading_reference = self.walk_heading
        gate = float(np.clip((.68 - abs(error)) / .50, .28, 1.))
        desired_speed = speed * gate
        self.forward_command = min(desired_speed, self.forward_command + .002)
        self.phase_rate = 1. + (rate - 1.) * float(np.clip((self.forward_command - .063) / .04, 0., 1.))
        return MotionCommand(self.forward_command, 0., 0., 1., "p070")

    def point(self, pose, dx, dy, speed, rate=1.):
        return self.steer(pose[2], math.atan2(dy, dx), speed, rate)

    def side_center(self, pose):
        """Estimate the track ahead from the two observed walls."""
        x, y, _ = pose
        centers = []
        for dx in (.19, .31, .43):
            ahead = self.hits[np.abs(self.hits[:, 0] - x - dx) <= .057, 1]
            lower = ahead[ahead < y - .11]
            upper = ahead[ahead > y + .11]
            if lower.size and upper.size:
                centers.append((lower.max() + upper.min()) / 2.)
        if centers:
            drift = float(np.median(centers)) - self.centerline_y
            self.centerline_y += float(np.clip(.28 * drift, -.027, .027))
        return self.centerline_y

    def sample_grade(self, pose):
        if not self.rough_map:
            return
        x, y, yaw = pose
        height = float(A.data.qpos[2]) + .26
        dx, dy = .17 * math.cos(yaw), .17 * math.sin(yaw)
        near = ground_height(x - dx, y - dy, height)
        far = ground_height(x + dx, y + dy, height)
        if near is not None and far is not None:
            self.grade = math.degrees(math.atan2(far - near, .34))
        fore = ground_height(x + .28, y, height)
        back = ground_height(x, y, height)
        if fore is not None and back is not None:
            self.along_grade = math.degrees(math.atan2(fore - back, .28))

    def corridor_speed(self, pose, t, scan):
        if scan:
            self.sample_grade(pose)
        grade = self.along_grade if abs(self.along_grade) > abs(self.grade) else self.grade
        if grade > 2.5:
            speed, rate, surface = .056, 1., "uphill"
        elif grade < -2.5:
            speed, rate, surface = .037, 1., "downhill"
        else:
            speed, rate, surface = .071, 1., "level"
        if surface != self.last_surface:
            self.event("surface_mode", t, pose, mode=surface, measured_grade_deg=grade)
            self.last_surface = surface
        return speed, rate

    def command(self, pose, t, hits=None):
        x, y, yaw = pose
        scan = hits is not None
        if scan:
            self.hits = hits
        self.phase_rate = 1.
        self.heading_reference = None

        if self.phase == "approach":
            front = self.hits[(self.hits[:, 0] > x) & (abs(self.hits[:, 1]) < .115)]
            if scan and len(front) >= 3 and min(front[:, 0]) - x < .64:
                self.junction = x
                self.choose(t, pose)
            else:
                return self.point(pose, .55, -y, self.cruise, self.cruise_rate)
        if self.phase in ("hold", "settle"):
            return brake()
        lane = self.lanes[self.branch]

        if self.phase == "align":
            dy = lane - y
            if abs(dy) < .065 and abs(wrap(yaw)) < .46:
                self.phase = "explore"
                choice = next(e for e in reversed(self.events) if e["event"] == "choose_branch")
                self.event("enter_branch", t, pose, branch=self.branch, selection_pose=choice["pose"])
            else:
                look = float(np.clip(self.entry_x - x, .10, .28))
                target_yaw = math.atan2(dy, look)
                return self.steer(yaw, target_yaw, .064 if abs(dy) > .07 else .046)

        if self.phase == "explore":
            if scan:
                offset = self.hits[:, 0] - x
                forward = self.hits[(offset > 0) & (offset < .34)
                                    & (abs(self.hits[:, 1] - lane) < .10)]
                toward_center = (lane - self.hits[:, 1]) * math.copysign(1., lane)
                inner = self.hits[(toward_center > .12) & (toward_center < .45)]
                beside = bool(np.any(abs(inner[:, 0] - x) <= .10))
                self.beside_divider |= beside
                if len(forward) >= 4:
                    self.blocked.add(self.branch)
                    self.blocked_pose = pose
                    self.turn_goal = math.pi
                    self.phase = "turn_back"
                    self.forward_command = 0.
                    self.pivoting = False
                    self.walk_heading = None
                    self.event("dead_end_detected", t, pose, branch=self.branch,
                               obstacle_range_m=float(min(forward[:, 0]) - x),
                               hits_in_corridor=len(forward))
                elif self.beside_divider and not np.any(abs(inner[:, 0] - x) <= .19):
                    self.phase = "finish"
                    self.centerline_y = y
                    self.event("clear_of_divider", t, pose, branch=self.branch)
            if self.phase == "explore":
                return self.point(pose, .39, lane - y, self.branch_speed, self.branch_rate)

        if self.phase == "turn_back":
            error = wrap(self.turn_goal - yaw)
            if abs(error) > .14:
                turn = math.copysign(float(np.clip(2.7 * abs(error), .13, self.turn_rate)), error)
                return MotionCommand(0., 0., turn, 1., "turn")
            self.phase = "retreat"
            self.event("turned_around", t, pose, branch=self.branch,
                       heading_change_deg=abs(math.degrees(wrap(yaw - self.blocked_pose[2]))))

        if self.phase == "retreat":
            if x > self.junction + .035:
                return self.point(pose, -.39, lane - y, .094, 1.08)
            self.phase = "turn_forward"
            self.event("returned_to_junction", t, pose, branch=self.branch,
                       retreat_m=self.blocked_pose[0] - x)
        if self.phase == "turn_forward":
            self.forward_command = 0.
            if scan:
                self.choose(t, pose)
            return brake()

        if self.phase == "finish":
            dx, dy = self.goal - np.array([x, y])
            distance = math.hypot(dx, dy)
            if distance <= .065:
                self.phase = "settle"
                self.event("entered_goal", t, pose)
                return brake()
            speed, rate = self.corridor_speed(pose, t, scan)
            if not self.rough_map:
                goal_speed = float(np.clip(.48 * distance, .048, .092))
                return self.point(pose, dx, dy, goal_speed)
            center = self.side_center(pose) if scan else self.centerline_y
            desired = float(np.clip(math.atan2(center - y, .46), -.42, .42))
            if distance < .60:
                weight = float(np.clip((.60 - distance) / .49, 0., 1.))
                goal_yaw = math.atan2(dy, max(dx, .09))
                desired += weight * wrap(goal_yaw - desired)
                speed = min(speed, .059)
            return self.steer(yaw, desired, speed, rate, turn_limit=.24)
        return brake()
