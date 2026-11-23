"""Check the public MiniDuck showcase or replay the maze with matching local inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
MAZE = WORK / "miniduck_fixed_maze"
NATIVE = WORK / "miniduck_native_gate_20261123"
PREDICTIVE = WORK / "slope10_final_candidate_v1_20261123"
XML_LAYOUT = Path("training/adapted/playground/open_duck_mini_v2/xmls")
POLICY_LAYOUT = Path("models/duck_initial")
CHECKS = (
    "training/classroom_paths.py",
    "training/portable_policy.py",
    "training/terrain_course.py",
    "deployment/mini_bdx_runtime/mini_bdx_runtime/mujoco_heading_policy.py",
    str(XML_LAYOUT / "scene_flat_terrain_no_head.xml"),
)
ASSETS = (
    str(POLICY_LAYOUT / "slope6_ppo_p3_128k.npz"),
    str(POLICY_LAYOUT / "accepted_6deg_final.json"),
)


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check():
    summary = json.loads((ROOT / "results/evaluation-summary.json").read_text(encoding="utf-8"))
    video = summary["full_maze_video"]
    for path, expected in (
        (ROOT / video["file"], video["sha256"]),
        (MAZE / "map_center_exit.json", summary["full_maze"]["map_sha256"]),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise SystemExit(f"Missing or mismatched public file: {path.relative_to(ROOT)}")
    goal = summary["full_maze"]
    trials = goal["cases"]
    if (len(trials) != goal["completed_trials"]
            or sum(row["full_pass"] for row in trials) != goal["passed"]
            or not any(row["seed"] == video["seed"] and row["full_pass"] for row in trials)):
        raise SystemExit("The public maze results are internally inconsistent.")
    print("Public maze map, video integrity, and saved summary checked.")


def runtime_environment(classroom, runtime_assets):
    classroom = classroom.expanduser().resolve()
    runtime_assets = runtime_assets.expanduser().resolve()
    missing = [classroom / part for part in CHECKS if not (classroom / part).is_file()]
    missing.extend(runtime_assets / part for part in ASSETS if not (runtime_assets / part).is_file())
    if missing:
        raise SystemExit("Missing local MiniDuck inputs. See README.md:\n" + "\n".join(map(str, missing)))

    paths = (WORK, NATIVE, PREDICTIVE, classroom / "training", MAZE)
    env = os.environ.copy()
    env.update(
        MINIDUCK_ROOT=str(runtime_assets),
        MINIDUCK_TRAINING=str(classroom),
        MINIDUCK_ASSET_ROOT=str(classroom),
        MINIDUCK_XML_DIR=str(classroom / XML_LAYOUT),
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
    )
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("check", help="Check the published map, video and result summary")
    maze = modes.add_parser("maze", help="Run the final closed-maze recovery controller")
    render = modes.add_parser("render", help="Render a previously saved maze trace at 1×")
    for mode in (maze, render):
        mode.add_argument("--classroom", required=True, type=Path)
        mode.add_argument("--runtime-assets", required=True, type=Path)
    maze.add_argument("--out", required=True, type=Path)
    maze.add_argument("--seeds", nargs="+", type=int, default=[105391])
    maze.add_argument("--exit-only", action="store_true", help="Run only the exit and dance")
    render.add_argument("--run", required=True, type=Path)
    render.add_argument("--seed", type=int, default=105391)
    args = parser.parse_args()

    if args.mode == "check":
        check()
        return 0

    env = runtime_environment(args.classroom, args.runtime_assets)
    if args.mode == "maze":
        output = args.out.expanduser().resolve()
        if output.exists():
            parser.error(f"Choose a new output directory: {output}")
        script = MAZE / "center_exit_recovery.py"
        forwarded = ["--out", str(output), "--seeds", *map(str, args.seeds)]
        if args.exit_only:
            forwarded.append("--exit-only")
    else:
        run = args.run.expanduser().resolve()
        script = MAZE / "render.py"
        forwarded = [str(run), "normal", str(args.seed), "--trace", f"center_exit_{args.seed}"]
    return subprocess.call([sys.executable, str(script), *forwarded], env=env)


if __name__ == "__main__":
    sys.exit(main())
