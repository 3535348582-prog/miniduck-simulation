"""Limit downhill momentum before handing the last slope back to flat cruise."""

from dataclasses import asdict, dataclass
import math

from center_exit_adaptive import AdaptiveCenterPilot
from center_exit_test import A, main, np
from slope_preview import BalancePreview


@dataclass(frozen=True)
class RecoverySettings:
    forward_limit: float
    progress_weight: float
    command_cap: float


RECOVERY_SETTINGS = {
    "matched_speed": RecoverySettings(.105, 5.0, .046),
    "lower_momentum": RecoverySettings(.090, 3.8, .046),
    "slow_steps": RecoverySettings(.110, 5.0, .040),
}


class ExitBalancePreview(BalancePreview):
    def active_on_grade(self):
        return self.nav.recovery_active or super().active_on_grade()


class RecoveringCenterPilot(AdaptiveCenterPilot):
    recovery_settings = RECOVERY_SETTINGS["matched_speed"]
    preview_forward_limit = .22
    preview_progress_weight = 8.

    def __init__(self, goal, profile):
        super().__init__(goal, profile)
        self.recovery_active = False
        self.recovery_completed = False
        self.recovery_heading = None
        self.recovery_stable_since = None

    def configure_policy(self, actor):
        self.preview = ExitBalancePreview(actor, self)
        if self.use_preview:
            self.preview.sequence_search = self.sequence_driving
            actor.infer = self.preview

    def flat_drive_ready(self):
        return super().flat_drive_ready() and not self.recovery_active

    def steer(self, yaw, desired, speed, rate=1., turn_limit=None):
        data = A.data
        pose = (*map(float, data.qpos[:2]), yaw)
        t = float(data.time)
        distance = math.dist(pose[:2], self.goal)
        west_locked = self.grade_direction is not None and self.grade_direction < 0.
        steep_downhill = self.last_surface == "downhill" or (
            self.forward_grade is not None and self.forward_grade < -6.)
        if not self.recovery_active and not self.recovery_completed and west_locked:
            if self.approach_grade >= 8.5 and distance < 2.54 and steep_downhill:
                self.recovery_active = True
                self.preview_forward_limit = self.recovery_settings.forward_limit
                self.preview_progress_weight = self.recovery_settings.progress_weight
                self.preview.remaining = 0
                self.event("steep_exit_recovery_on", t, pose,
                           observed_grade_deg=self.grade,
                           peak_approach_grade_deg=self.approach_grade,
                           forward_limit_m_s=self.preview_forward_limit)
        if self.recovery_active:
            rot = data.site_xmat[A.imu].reshape(3, 3)
            tilt = math.degrees(math.acos(float(np.clip(rot[2, 2], -1., 1.))))
            roll = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
            if west_locked:
                self.recovery_heading = math.pi
            stable = (self.flat_confirmed and tilt < 43. and abs(roll) < 21.
                      and np.linalg.norm(data.qvel[:2]) < .20
                      and np.linalg.norm(data.qvel[3:5]) < 3.2
                      and np.any(A.contacts()))
            if stable:
                if self.recovery_stable_since is None:
                    self.recovery_stable_since = t
            else:
                self.recovery_stable_since = None
            if self.recovery_stable_since is not None and t - self.recovery_stable_since >= .18:
                self.recovery_active = False
                self.recovery_completed = True
                self.preview_forward_limit = .22
                self.preview_progress_weight = 8.
                self.event("steep_exit_recovery_off", t, pose, tilt_deg=tilt, roll_deg=roll)
            else:
                speed = min(speed, self.recovery_settings.command_cap)
                if not west_locked and self.recovery_heading is not None:
                    desired = self.recovery_heading
                rate = 1.
        return super().steer(yaw, desired, speed, rate, turn_limit)

    def speed_report(self):
        report = super().speed_report()
        report["steep_exit_recovery"] = {
            "settings": asdict(self.recovery_settings),
            "completed": self.recovery_completed,
            "events": [event for event in self.events if event["event"].startswith("steep_exit_recovery_")],
        }
        return report


if __name__ == "__main__":
    report = main(pilot_type=RecoveringCenterPilot)
    counts = report["totals"]
    passed = counts["danced"] if report["scope"] == "exit_and_dance_component" else counts["full"]
    if passed != counts["trials"]:
        raise SystemExit(1)
