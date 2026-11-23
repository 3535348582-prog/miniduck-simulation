"""Configurable goal interfaces. Coordinate adapters are not visual perception."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GoalSpec:
    mode: str
    marker_id: int = 0
    marker_side_length_m: float = 0.20
    target_stop_offset_m: tuple = (0.0, 0.0, 0.30)
    arrival_radius_m: float = 0.10


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class GoalEstimate:
    timestamp: float
    relative_xy_m: tuple
    bearing_rad: float
    uncertainty_m: float
    valid: bool
    source: str


def _rotate_inv(dx, dy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * dx + s * dy, -s * dx + c * dy)


def _rotate(dx, dy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (c * dx - s * dy, s * dx + c * dy)


class CoordinateGoalAdapter:
    def __init__(self, spec, target_xy):
        if spec.mode != "explicit_coordinate":
            raise ValueError("CoordinateGoalAdapter requires explicit_coordinate mode")
        self.spec = spec
        self.target_xy = (float(target_xy[0]), float(target_xy[1]))

    def estimate(self, pose, timestamp):
        dx = self.target_xy[0] - pose.x
        dy = self.target_xy[1] - pose.y
        rel = _rotate_inv(dx, dy, pose.yaw)
        return GoalEstimate(
            timestamp=float(timestamp),
            relative_xy_m=rel,
            bearing_rad=math.atan2(rel[1], rel[0]),
            # Explicit coordinate oracle uses the exact simulated pose/goal.
            # arrival_radius_m is an acceptance tolerance, not a measurement
            # uncertainty. Real localization errors must be supplied separately.
            uncertainty_m=0.0,
            valid=True,
            source="explicit_coordinate",
        )


class SimulatedMarkerGoalAdapter:
    def __init__(self, spec, marker_pose, noise_m=0.0, max_range_m=8.0, seed=0):
        if spec.mode != "visual_marker":
            raise ValueError("SimulatedMarkerGoalAdapter requires visual_marker mode")
        self.spec = spec
        self.marker_pose = marker_pose
        self.noise_m = float(noise_m)
        self.max_range_m = float(max_range_m)
        self._state = int(seed) & 0xffffffff

    def _u(self):
        self._state = (1103515245 * self._state + 12345) & 0x7fffffff
        return self._state / float(0x7fffffff)

    def estimate(self, pose, timestamp):
        marker = self.marker_pose
        stop = _rotate(self.spec.target_stop_offset_m[0],
                       self.spec.target_stop_offset_m[1], marker.yaw)
        tx, ty = marker.x + stop[0], marker.y + stop[1]
        dx, dy = tx - pose.x, ty - pose.y
        rng = math.hypot(dx, dy)
        if rng > self.max_range_m:
            return GoalEstimate(float(timestamp), (0.0, 0.0), 0.0,
                                float("inf"), False, "simulated_marker_out_of_range")
        rel = _rotate_inv(dx, dy, pose.yaw)
        if self.noise_m:
            rel = (rel[0] + (2 * self._u() - 1) * self.noise_m,
                   rel[1] + (2 * self._u() - 1) * self.noise_m)
        return GoalEstimate(float(timestamp), rel, math.atan2(rel[1], rel[0]),
                            float(self.noise_m), True,
                            "simulated_visual_marker_oracle")
