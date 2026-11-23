"""Deterministic turn-then-walk mission FSM; no hidden goal state is read."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MotionCommand:
    vx: float
    vy: float
    yaw_rate: float
    tempo_scale: float = 1.0
    mode: str = "hold"


@dataclass(frozen=True)
class MissionStatus:
    state: str
    nav_success: bool
    dance_success: bool
    mission_success: bool
    failure_reason: str = None


class MissionFSM:
    def __init__(self, arrival_radius_m=0.10, settle_s=2.0, dance_s=5.0,
                 hold_s=2.0, turn_threshold_rad=0.25,
                 approach_margin_m=None, acceptance_radius_m=None,
                 cruise_speed_m_s=0.06):
        # Keep the old arrival_radius_m alias for callers that only know one
        # radius. New callers pass the control margin and acceptance radius
        # separately; SETTLE always uses the acceptance radius.
        self.acceptance_radius_m = float(arrival_radius_m if acceptance_radius_m is None else acceptance_radius_m)
        self.approach_margin_m = float(arrival_radius_m if approach_margin_m is None else approach_margin_m)
        self.arrival_radius_m = self.approach_margin_m
        self.settle_s = float(settle_s)
        self.dance_s = float(dance_s)
        self.hold_s = float(hold_s)
        self.turn_threshold_rad = float(turn_threshold_rad)
        self.cruise_speed_m_s = float(cruise_speed_m_s)
        if not (0.0 < self.cruise_speed_m_s <= 1.0):
            raise ValueError("cruise_speed_m_s must be in (0, 1]")
        self.state = "NAVIGATE"
        self.nav_success = False
        self.dance_success = False
        self.mission_success = False
        self.settle_elapsed = 0.0
        self.dance_elapsed = 0.0
        self.hold_elapsed = 0.0
        self.settle_ticks = 0
        self.dance_ticks = 0
        self.hold_ticks = 0
        self.failure_reason = None

    def _status(self):
        return MissionStatus(self.state, self.nav_success, self.dance_success,
                             self.mission_success, self.failure_reason)

    def _hold(self, mode):
        return MotionCommand(0.0, 0.0, 0.0, 1.0, mode), self._status()

    def _settle_ok_estimate(self, estimate, dist):
        if estimate is None or not bool(getattr(estimate, 'valid', False)):
            return False
        ts = float(getattr(estimate, 'timestamp', float('nan')))
        if not math.isfinite(ts):
            return False
        uncertainty = float(getattr(estimate, 'uncertainty_m', 0.0))
        if not math.isfinite(uncertainty) or uncertainty < 0.0:
            return False
        # The 8 cm value is a control margin, not the acceptance gate. The
        # external acceptance radius remains caller-supplied and unchanged.
        return dist + uncertainty <= self.acceptance_radius_m

    def step(self, estimate, robot, dt):
        if estimate is None or not estimate.valid:
            self.state = "SAFE_HOLD"
            self.failure_reason = self.failure_reason or "goal_estimate_invalid"
            return self._hold("safe_hold")
        x, y = estimate.relative_xy_m
        dist = math.hypot(x, y)
        bearing = math.atan2(y, x)
        speed = float(getattr(robot, "speed", 0.0))
        yaw_rate = float(getattr(robot, "yaw_rate", 0.0))
        angular = float(getattr(robot, "angular_speed", abs(yaw_rate)))
        arrived = (self._settle_ok_estimate(estimate, dist)
                   and speed < 0.02 and abs(yaw_rate) < 0.10 and angular < 0.20)
        if self.state in ("NAVIGATE", "APPROACH"):
            if arrived:
                self.state = "SETTLE"
                self.settle_elapsed = 0.0
                self.settle_ticks = 0
                return self._hold("settle")
            if dist < 0.50 and abs(y) > 0.04:
                # Narrow terminal guard only; the earlier all-distance brake
                # candidate caused approach timeouts.
                if dist < 0.15 and speed >= 0.08:
                    return self._hold("brake")
                return (MotionCommand(0.0, max(-0.08, min(0.08, 1.2 * y)),
                                      0.0, 1.0, "strafe"), self._status())
            if dist <= self.approach_margin_m * 1.5:
                if speed < 0.02 and dist > self.approach_margin_m:
                    # Use the signed relative goal, not the scalar distance.
                    # A target behind the robot must not receive positive creep.
                    if abs(x) >= abs(y):
                        vx = max(-0.03, min(0.03, 0.35 * x))
                        return (MotionCommand(vx, 0.0, 0.0, 1.0, "creep"),
                                self._status())
                    vy = max(-0.03, min(0.03, 0.35 * y))
                    return (MotionCommand(0.0, vy, 0.0, 1.0, "strafe"),
                            self._status())
                return self._hold("brake")
            if abs(bearing) > self.turn_threshold_rad:
                return (MotionCommand(0.0, 0.0, max(-0.35, min(0.35, 1.5 * bearing)),
                                      1.0, "turn"), self._status())
            return (MotionCommand(max(0.03, min(self.cruise_speed_m_s, 0.35 * dist)),
                                  0.0, 0.0, 1.0, "walk"), self._status())
        if self.state == "SETTLE":
            settled = (self._settle_ok_estimate(estimate, dist)
                       and speed < 0.02 and abs(yaw_rate) < 0.10 and angular < 0.20)
            if not settled:
                self.settle_elapsed = 0.0
                self.settle_ticks = 0
                self.state = "APPROACH"
                return self._hold("re-approach")
            self.settle_ticks += 1
            self.settle_elapsed = self.settle_ticks * dt
            needed = max(1, int(round(self.settle_s / dt)))
            if self.settle_ticks >= needed:
                self.nav_success = True
                self.state = "VICTORY"
                self.dance_elapsed = 0.0
                self.dance_ticks = 0
            return self._hold("settle")
        if self.state == "VICTORY":
            self.dance_ticks += 1
            self.dance_elapsed = self.dance_ticks * dt
            needed = max(1, int(round(self.dance_s / dt)))
            if self.dance_ticks >= needed:
                self.dance_success = True
                self.state = "HOLD"
                self.hold_elapsed = 0.0
                self.hold_ticks = 0
            return self._hold("victory")
        if self.state == "HOLD":
            self.hold_ticks += 1
            self.hold_elapsed = self.hold_ticks * dt
            needed = max(1, int(round(self.hold_s / dt)))
            if self.hold_ticks >= needed:
                self.mission_success = self.nav_success and self.dance_success
                self.state = "DONE" if self.mission_success else "SAFE_HOLD"
            return self._hold("hold")
        return self._hold("safe_hold")
