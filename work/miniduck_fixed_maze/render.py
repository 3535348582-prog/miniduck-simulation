"""Render a measured run on the verified maze, using the recorded native poses."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

from evaluate import HERE, INVOCATION_DIR, MAP, PHASES, build_world, mujoco, np, sha


LABELS = {
    "approach": "前进测距",
    "align": "选择通道",
    "explore": "沿通道前进",
    "turn_back": "发现死胡同，原地掉头",
    "retreat": "走回岔口",
    "turn_forward": "转回前进方向",
    "finish": "离开障碍区，接近终点",
    "settle": "进入终点区，停稳",
    "hold": "无可通行分支，停车",
}
SLOPE_COLORS = {
    "flat": (112, 127, 148),
    "up": (247, 126, 40),
    "crest": (254, 217, 84),
    "down": (42, 180, 238),
}


def color_slopes(spec, layout):
    """Color the visible surface of the existing ramps and heightfield."""
    terrain = layout.get("terrain", {})
    bands = terrain.get("grade_bands", ())
    hills = terrain.get("rounded_hills", ())
    if not bands and not hills:
        return
    if terrain.get("planar_ramps"):
        for i, band in enumerate(bands):
            parts = {"up": "up", "crest": "crest", "down": "down"}
            if band.get("toe_apron"):
                parts.update(entry_apron="up", exit_apron="down")
            for part, color in parts.items():
                spec.geom(f"grade_{i}_{part}").rgba = [*(v / 255 for v in SLOPE_COLORS[color]), 1.]
        return

    x0, x1 = terrain["x_bounds"]
    y0, y1 = terrain["y_bounds"]
    xs = np.linspace(x0, x1, 768)
    ys = np.linspace(y0, y1, 768)
    pixels = np.empty((len(ys), len(xs), 3), dtype=np.uint8)
    pixels[:] = SLOPE_COLORS["flat"]
    for shape in (*bands, *hills):
        along = shape["direction"] * (xs - shape["start_x"])
        if "length" in shape:
            length = shape["length"]
            rise_end, crest_end = .45 * length, .55 * length
        else:
            rise_end = shape["ascent"]
            crest_end = rise_end + shape["crest"]
            length = crest_end + shape["descent"]
        lo, hi = shape["y_bounds"]
        across = (ys >= lo) & (ys <= hi)
        pixels[np.ix_(across, (along >= 0.) & (along < rise_end))] = SLOPE_COLORS["up"]
        pixels[np.ix_(across, (along >= rise_end) & (along < crest_end))] = SLOPE_COLORS["crest"]
        pixels[np.ix_(across, (along >= crest_end) & (along <= length))] = SLOPE_COLORS["down"]
        edges = (np.abs(along) < .017) | (np.abs(along - length) < .017)
        pixels[np.ix_(across, edges)] = (240, 245, 251)

    name = "replay_terrain_colors"
    spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D,
                     width=len(xs), height=len(ys), nchannel=3, data=pixels.ravel())
    material = spec.add_material(name=name, texuniform=0, texrepeat=[1., 1.], shininess=0., specular=0.)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    ground = spec.geom("rough_ground")
    ground.material = name
    ground.rgba = [1., 1., 1., 1.]


def paint_speed_bumps(scene, layout):
    """Make the actual raised sections legible from the replay camera."""
    terrain = layout["terrain"]
    profile = np.asarray(terrain["x_height"])
    for bump in terrain.get("speed_bumps", ()):
        low, high = bump["x_bounds"]
        for fraction in np.linspace(.08, .92, 13):
            x = low + fraction * (high - low)
            height = float(np.interp(x, profile[:, 0], profile[:, 1])
                           + bump["height"] * np.sin(np.pi * fraction) ** 2)
            edges = []
            for side, default in (("lower", -.71), ("upper", .71)):
                path = np.asarray(layout["wall_paths"][side])
                edges.append(float(np.interp(x, path[:, 0], path[:, 1])) if x >= path[0, 0] else default)
            y0, y1 = edges
            color = [.93, .64, .13, 1.] if int(fraction * 20) % 4 < 3 else [.25, .23, .17, 1.]
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                np.array([.035 * (high - low), (y1 - y0) / 2 - .035, .0008]),
                np.array([x, (y1 + y0) / 2, height + .0022]),
                np.eye(3).ravel(), np.array(color, dtype=np.float32))
            scene.ngeom += 1


def render(folder, side, seed, controller=None, gate_layout=None, output_folder=None, trace_name=None):
    data = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if gate_layout is not None:
        from blind_test import MAP as MULTIFORK_MAP, physical_layout

        map_path = MULTIFORK_MAP
    else:
        map_path = HERE / data.get("map_file", MAP.name)
    if data["map_sha256"] != sha(map_path):
        raise ValueError("The archived run uses a different map")
    if trace_name is not None:
        row = next(row for row in data["rows"] if Path(row["trace_file"]).stem == trace_name)
    else:
        row = next(row for row in data["rows"] if row["seed"] == seed
                   and row["preferred_when_equal"] == side and row.get("controller") == controller
                   and row.get("layout") == gate_layout)
    layout = json.loads(map_path.read_text(encoding="utf-8"))
    if gate_layout is not None:
        layout = physical_layout(layout, gate_layout)
    motion_lab = layout.get("map_id") == "miniduck_motion_and_curved_slopes_v6"
    if motion_lab and "action" not in row:
        layout["goal_xy"] = row.get("goal_xy", layout["stations"][row["station"]]["goal_xy"])
    with np.load(folder / row["trace_file"], allow_pickle=False) as trace:
        poses, phases, path = trace["qpos"], trace["phase"], trace["path"]
        show_gait = (motion_lab and "action" not in row) or "adaptive_speed" in row
        gait = trace["gait"] if show_gait and "gait" in trace else None
        dance = row.get("celebration")
        if dance and dance["dance_ticks"]:
            extra = trace["dance_qpos"]
            first_time = path[-1, 0] + .04
            offset = round((first_time - row["seconds"]) / .02) - 1
            extra = extra[max(0, offset)::2]
            times = first_time + .04 * np.arange(len(extra))
            extra_path = np.column_stack((times, extra[:, :2], np.zeros(len(extra))))
            poses = np.concatenate((poses, extra))
            path = np.concatenate((path, extra_path))
            phases = np.concatenate((phases, np.full(len(extra), PHASES.index("settle"))))
    if len(poses) != len(phases) or len(poses) != len(path) or not np.allclose(np.diff(path[:, 0]), .04):
        raise ValueError("Expected an unbroken 25 Hz pose trace")
    model = build_world(layout, decorate=color_slopes)
    state = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    rough = "terrain" in layout
    colored_slopes = any(layout.get("terrain", {}).get(key) for key in ("grade_bands", "rounded_hills"))
    static_course = bool(data.get("fixed_layout") and layout.get("checkpoints"))
    center_exit = "exit" in layout
    if motion_lab:
        camera.distance, camera.azimuth, camera.elevation = 2.22, 115., -53.
    elif center_exit:
        camera.distance, camera.azimuth, camera.elevation = 2.42, 113., -53.
    elif static_course:
        camera.distance, camera.azimuth, camera.elevation = 2.10, 116., -58.
    elif gate_layout is not None:
        camera.lookat[:] = [3.42, 0., .05]
        camera.distance, camera.azimuth, camera.elevation = 7.14, 90., -71.
    else:
        camera.lookat[:] = [2.54 if rough else 1.30, 0., .06]
        camera.distance, camera.azimuth, camera.elevation = (5.30 if rough else 3.52), 90., (-61. if rough else -78.)
    width, height = 800, 450
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[4] = 1
    if rough:
        scene_option.geomgroup[3] = 1
    font = os.environ.get("MINIDUCK_FONT", "C:/Windows/Fonts/msyh.ttc")
    try:
        heading, detail = ImageFont.truetype(font, 25), ImageFont.truetype(font, 17)
    except OSError:
        heading, detail = ImageFont.load_default(size=25), ImageFont.load_default(size=17)

    output = (output_folder or folder) / f"{Path(row['trace_file']).stem}_1x.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    events = row["events"]
    choices = [event for event in events if event["event"] == "choose_branch"]
    blocked = next((event for event in events if event["event"] in
                    ("dead_end_detected", "branch_rejected_from_scan")), None)
    surfaces = [event for event in events if event["event"] == "surface_mode"]
    pace_modes = [event for event in events if event["event"] == "pace_mode"]
    adaptive_maze_modes = row.get("adaptive_speed", {}).get("modes", [])
    bumps = [event for event in events if event["event"] == "raised_surface_detected"]
    sides = {"left": "左路", "right": "右路"}
    with mujoco.Renderer(model, width=width, height=height) as renderer, subprocess.Popen([
        ffmpeg, "-n", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], stdin=subprocess.PIPE) as encoder:
        for (t, x, y, _), qpos, phase in zip(path, poses, phases):
            state.qpos[:] = qpos
            mujoco.mj_fwdPosition(model, state)
            if center_exit or motion_lab:
                camera.lookat[:] = [x, y, .11 + .43 * qpos[2]]
            elif static_course:
                camera.lookat[:] = [x + .28, y, .15 + .4 * qpos[2]]
            renderer.update_scene(state, camera=camera, scene_option=scene_option)
            if static_course:
                paint_speed_bumps(renderer.scene, layout)
            if center_exit:
                exit_info = layout["exit"]
                y0, y1 = exit_info["y_bounds"]
                scene = renderer.scene
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                                    np.array([.018, (y1-y0)/2, .001]),
                                    np.array([exit_info["x"], (y1+y0)/2, .002]),
                                    np.eye(3).ravel(), np.array([.15, .95, .50, .75], dtype=np.float32))
                scene.ngeom += 1
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, width, 76), fill="#111827")
            stage = LABELS[PHASES[phase]]
            if controller == "heading" and PHASES[phase] == "align":
                stage = "转向进入通道"
            ground = [e["mode"] for e in surfaces if e["time_s"] <= t]
            terrain_title = {"level": "沿弯道前进", "uphill": "上坡", "downhill": "下坡"}
            if rough and ground and PHASES[phase] == "finish":
                stage = terrain_title[ground[-1]]
            if gate_layout is not None:
                if PHASES[phase] == "explore":
                    stage = terrain_title.get(ground[-1], "测距并规划") if ground else "测距并规划"
                    if stage == "沿弯道前进":
                        stage = "测距并规划"
                stage = f"摆法 {gate_layout} · {stage}"
            if static_course:
                if PHASES[phase] == "explore":
                    stage = terrain_title.get(ground[-1], "测距并规划") if ground else "测距并规划"
                    if stage == "沿弯道前进":
                        stage = "测距并规划"
                    if any(0 <= t - e["time_s"] < 5.8 for e in bumps):
                        stage = "已探测到前方低凸起"
                stage = f"固定路障课 · {stage}"
            if center_exit:
                center_stage = {"exit_and_dance_component": "出口直道 · 接近终点",
                                "native_recovery_tail": "相同下坡起点 · 坡脚复测"}
                stage = center_stage.get(data.get("scope"), "中心起步 · 测距并选路")
                for band in layout["terrain"]["grade_bands"]:
                    u = (x - band["start_x"]) * band["direction"]
                    up, crest, down = band["ascent"], band["crest"], band["descent"]
                    if 0 <= u <= up + crest + down and band["y_bounds"][0] <= y <= band["y_bounds"][1]:
                        segment = "上坡" if u < up else "坡顶" if u < up + crest else "下坡"
                        stage = f"{band['degrees']}° {segment}"
                        break
                crossing = dance.get("finish_crossing_s") if dance else None
                if crossing is not None and t >= crossing:
                    stage = "已出迷宫 · 到达终点"
                maze_pace = next((event for event in reversed(adaptive_maze_modes)
                                  if event["time_s"] <= t), None)
                if maze_pace is not None:
                    caption = {"flat_cruise": "检测到平地 · 巡航",
                               "approach_slope": "识别前方坡道 · 减速",
                               "finish_sprint": "终点平地 · 冲刺",
                               "finish_brake": "终点前制动"}
                    stage = caption.get(maze_pace["mode"], stage)
                    if PHASES[phase] == "settle":
                        stage = "到达并停稳"
                if dance and dance["dance_ticks"] and t >= row["seconds"]:
                    dancing = t < row["seconds"] + dance["dance_duration_s"]
                    stage = "胜利舞" if dancing else "舞后停稳"
            if motion_lab:
                station_label = {"flat": "平地", "low": "5°坡", "arch": "倒 U 形坡",
                                 "wave": "连续波浪坡", "inclined_pad": "5°动作台"}
                actions = {"forward": "前进", "backward": "后退", "left": "向左移动",
                           "right": "向右移动", "rotate_left": "原地左转", "rotate_right": "原地右转"}
                movement = actions.get(row.get("action"), terrain_title.get(ground[-1], "测试") if ground else "直行")
                active_pace = next((event for event in reversed(pace_modes) if event["time_s"] <= t), None)
                if active_pace is not None:
                    caption = {"cruise": "平地快速", "approach_slope": "发现坡道 · 提前减速",
                               "confirm_level": "确认平地", "sprint": "终点前冲刺", "brake": "终点制动"}
                    movement = caption.get(active_pace["mode"], movement)
                if PHASES[phase] == "settle":
                    movement = "到达并停稳"
                crest_started = next((e["time_s"] for e in events if e["event"] == "crest_brake"), math.inf)
                crest_resumed = next((e["time_s"] for e in events if e["event"] == "crest_resume"), math.inf)
                if crest_started <= t < crest_resumed:
                    movement = "坡顶停稳"
                stage = station_label[row["station"]] + " · " + movement
            draw.text((18, 7), stage, font=heading, fill="#f9fafb")
            current = [sides[event["branch"]] for event in choices if event["time_s"] <= t]
            memory = sides[blocked["branch"]] if blocked is not None and t >= blocked["time_s"] else "无"
            chosen = " → ".join(current) if current else "尚未选择"
            text = f"1× 仿真  {t:05.1f} 秒    ·    选路 {chosen}    ·    已排除 {memory}"
            if gate_layout is not None:
                crossings = [sides[e["side"]] for e in row["measured_crossings"] if e["time_s"] <= t]
                observed_route = " → ".join(crossings) or "未经过"
                text = f"1× 仿真  {t:05.1f} 秒    ·    实际过岔 {observed_route}"
            if static_course:
                passed_sides = [sides[e["side"]] for e in row["checkpoints"] if e["t_s"] <= t]
                route = " → ".join(passed_sides) or "未经过"
                text = f"1× 仿真  {t:05.1f} 秒    ·    实际通过方向 {route}"
            if center_exit:
                text = f"1× 仿真  {t:05.1f} 秒    ·    单出口封闭迷宫"
                if gait is not None and t < row["seconds"]:
                    speed = np.interp(t, gait[:, 0], gait[:, 2])
                    text += f"    ·    前进指令 {speed:.3f} m/s"
                    if maze_pace is not None and maze_pace["mode"] == "approach_slope":
                        angle = maze_pace["lookahead_grade_deg"]
                        if angle is not None:
                            text += f"    ·    前方 {angle:.1f}°"
                if row["fell"] and t >= row["seconds"] - .1:
                    text += "    ·    本次失稳"
            if motion_lab:
                text = f"1× 仿真  {t:05.1f} 秒"
                if gait is not None:
                    command_speed = np.interp(t, gait[:, 0], gait[:, 2])
                    old_t = max(path[0, 0], t - .60)
                    old_x = np.interp(old_t, path[:, 0], path[:, 1])
                    speed = abs(x - old_x) / max(.02, t - old_t)
                    text += f"    ·    前进实速 {speed:.3f} m/s    ·    指令 {command_speed:.3f} m/s"
                    if active_pace is not None and active_pace["mode"] == "approach_slope":
                        text += f"    ·    前方 {active_pace['ahead_grade_deg']:.1f}°"
                if row["fell"] and t >= row["seconds"] - .1:
                    text += "    ·    本次失稳"
            draw.text((18, 45), text, font=detail, fill="#b9c8db")
            if colored_slopes:
                for x0, color, label in ((622, "up", "上坡"), (681, "crest", "坡顶"), (740, "down", "下坡")):
                    draw.rectangle((x0, 12, x0 + 10, 22), fill=SLOPE_COLORS[color])
                    draw.text((x0 + 14, 6), label, font=detail, fill="#f9fafb")
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
    if encoder.returncode:
        raise RuntimeError(f"Encoding failed: {encoder.returncode}")
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(output), "-f", "null", "-"],
                   check=True)
    print(f"{output}  {len(path) / 25:.2f}s  sha256={sha(output)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("side", choices=["left", "right", "auto", "normal", "short", "off"])
    parser.add_argument("seed", type=int)
    parser.add_argument("--controller", choices=["heading", "baseline"])
    parser.add_argument("--gate-layout", choices=["A", "B", "C", "D"])
    parser.add_argument("--output-folder", type=Path)
    parser.add_argument("--trace")
    args = parser.parse_args()
    output_folder = (INVOCATION_DIR / args.output_folder).resolve() if args.output_folder else None
    render((INVOCATION_DIR / args.folder).resolve(), args.side, args.seed, args.controller,
           args.gate_layout, output_folder, args.trace)
