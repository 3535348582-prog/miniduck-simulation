"""Run the ten-second celebration from the actual native arrival state."""

from control.mission_runner import DT, MissionConfig, _dance_template
from slope10_predictive_snapshots_20261123 import A, np
from slope10_pace_sequence import physics_step


def celebrate(actor, mission_row, observe, *, amplitude=2.0, goal=(5., 0.),
              finish_line_x=5.0, direction=1, exit_y_bounds=None, sample_dt=DT):
    crossing = next((i for i, xy in enumerate(mission_row["path"])
                     if direction * (xy[0] - finish_line_x) >= 0.
                     and (exit_y_bounds is None or exit_y_bounds[0] <= xy[1] <= exit_y_bounds[1])), None)
    arrival_s = mission_row["seconds"]
    result = {
        "finish_line_x_m": finish_line_x,
        "finish_crossing_s": DT + crossing * sample_dt if crossing is not None else None,
        "arrival_s": arrival_s,
        "dance_ticks": 0,
        "post_dance_hold_s": 0.,
        "pass": False,
    }
    if crossing is None or not mission_row["strict_nav_success"]:
        return dict(result, reason="no_stable_finish_beyond_line")

    template = _dance_template(MissionConfig(dance=True, dance_s=10., dance_style="enhanced",
                                             dance_amplitude=amplitude, contract=actor.contract))
    limits = actor.limits
    allowed = float(actor.contract["max_motor_velocity"] * DT)
    final_hold = 0
    peak_tilt, max_delta = 0., 0.
    first_phases = {}
    start = np.asarray(A.data.qpos[:3]).copy()
    lowest_z, highest_z = start[2], start[2]
    worst_distance = float(np.linalg.norm(start[:2] - goal))
    latches = {"fell": False, "wall_hit": False, "body_ground_hit": False}
    for tick in range(template.intervals + 250):
        dancing = tick < template.intervals
        phase = "post_dance_hold"
        desired = actor.default
        if dancing:
            sample = template.sample(tick * DT)
            desired, phase = sample.target, sample.phase
            first_phases.setdefault(phase, arrival_s + tick * DT)
        last = A.data.ctrl.copy()
        desired = np.clip(desired, limits[:, 0], limits[:, 1])
        target = np.clip(desired, last - allowed, last + allowed)
        _, tilt, failed = physics_step(target)
        max_delta = max(max_delta, float(np.max(np.abs(target - last))))
        latches["fell"] |= bool(A.data.qpos[2] < .05 or tilt > 75.)
        latches["wall_hit"] |= bool(A.wall_hit())
        latches["body_ground_hit"] |= bool(A.body_ground_hit())
        observe(np.asarray(A.data.qpos).copy(), tilt)
        peak_tilt = max(peak_tilt, tilt)
        distance = float(np.linalg.norm(A.data.qpos[:2] - goal))
        worst_distance = max(worst_distance, distance)
        if dancing:
            result["dance_ticks"] += 1
            lowest_z = min(lowest_z, A.data.qpos[2])
            highest_z = max(highest_z, A.data.qpos[2])
        else:
            stable = (distance <= .10 and np.linalg.norm(A.data.qvel[:2]) < .02
                      and abs(A.data.qvel[5]) < .10 and np.linalg.norm(A.data.qvel[3:6]) < .20)
            final_hold = final_hold + 1 if stable else 0
        if failed or final_hold >= 100:
            break

    passed = (final_hold >= 100 and result["dance_ticks"] == template.intervals
              and worst_distance <= .10 and max_delta <= allowed + 1e-6 and not any(latches.values()))
    result.update(latches, post_dance_hold_s=final_hold * DT, peak_tilt_deg=peak_tilt,
                  amplitude=amplitude, body_height_range_m=float(highest_z - lowest_z),
                  max_arrival_distance_m=worst_distance, max_ctrl_delta_rad=max_delta,
                  dance_duration_s=result["dance_ticks"] * DT, phases=first_phases,
                  celebration_started_s=arrival_s,
                  ended_s=arrival_s + (tick + 1) * DT)
    result["pass"] = bool(passed)
    if not passed:
        result["reason"] = "post_finish_motion_failed_strict_checks"
    return result
