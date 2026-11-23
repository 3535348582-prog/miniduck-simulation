"""Single source of truth for the flat and slope courses.

The same XML files are used by native MuJoCo acceptance and by MJX training.
Walkable surfaces are floor + up_ramp + top_flat + down_ramp. Walls, when
present, are never part of the support/contact set.
"""
from pathlib import Path
import math
import os

import numpy as np

TRAINING_ROOT = Path(os.environ.get("MINIDUCK_TRAINING", Path(__file__).resolve().parents[2] / "miniducktraining_classroom"))
XML_DIR = Path(os.environ.get("MINIDUCK_XML_DIR", TRAINING_ROOT / "training/adapted/playground/open_duck_mini_v2/xmls"))
FLAT_XML = XML_DIR / "scene_flat_terrain_no_head.xml"
GROUND_NAMES = ("floor", "up_ramp", "top_flat", "down_ramp")
SLOPES = (5, 6, 6.5, 7, 7.5, 8, 10)

# x positions of flat/up/top/down/flat; height is piecewise linear.
X_POINTS = (-1.0, 0.8, 2.3, 2.7, 4.2, 6.0)


def height_points(deg):
    deg = float(deg)
    if deg <= 0:
        return X_POINTS, (0.0,) * len(X_POINTS)
    h = math.tan(math.radians(deg)) * 1.5
    return X_POINTS, (0.0, 0.0, h, h, 0.0, 0.0)


def height_np(x, deg):
    xp, zp = height_points(deg)
    return np.interp(np.asarray(x, dtype=np.float64), xp, zp)


def height_jax(deg):
    import jax.numpy as jp
    xp, zp = height_points(deg)
    xp, zp = jp.asarray(xp), jp.asarray(zp)
    return lambda x: jp.interp(x, xp, zp)


def _ramp_boxes(deg):
    a = math.radians(float(deg))
    h = math.tan(a) * 1.5
    length = math.hypot(1.5, h)
    t, w = .025, .70
    q_up = (math.cos(a / 2), 0., -math.sin(a / 2), 0.)
    q_dn = (math.cos(a / 2), 0., math.sin(a / 2), 0.)
    up_pos = (1.55 + math.sin(a) * t, 0., h / 2 - math.cos(a) * t)
    dn_pos = (3.45 - math.sin(a) * t, 0., h / 2 - math.cos(a) * t)
    f = lambda v: ' '.join(f'{x:.9f}' for x in v)
    return [
        ('up_ramp', f'{length / 2 + .03:.9f} {w:.2f} {t:.3f}', f(up_pos), f(q_up)),
        ('top_flat', f'0.2 {w:.2f} {t:.3f}', f'2.5 0 {h - t:.9f}', '1 0 0 0'),
        ('down_ramp', f'{length / 2 + .03:.9f} {w:.2f} {t:.3f}', f(dn_pos), f(q_dn)),
    ]


def xml_text(deg):
    boxes = _ramp_boxes(deg)
    geom_xml = []
    for name, size, pos, quat in boxes:
        geom_xml.append(
            f'        <geom name="{name}" type="box" size="{size}" pos="{pos}" quat="{quat}" '
            'friction="0.8 0.02 0.01" contype="1" conaffinity="1" rgba=".45 .55 .5 1"/>'
        )
    return f'''<mujoco model="scene_slope{deg}">
    <include file="open_duck_mini_v2_no_head.xml"/>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0"/>
    </asset>
    <worldbody>
        <body name="floor">
            <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" contype="1" conaffinity="0" priority="1" friction="0.6" condim="3"/>
        </body>
{chr(10).join(geom_xml)}
    </worldbody>
    <keyframe>
        <key name="home" qpos="
    0 0 0.140271261483
    0.974576417362 0 0.224055365304 0
    0 0 -0.399379879236 1.5 -0.623843669891
    0 0 0.399379879236 1.5 -0.623843669891
"
            ctrl="
    0 0 -0.399379879236 1.5 -0.623843669891
    0 0 0.399379879236 1.5 -0.623843669891
          "/>
    </keyframe>
</mujoco>
'''


def xml_path(deg):
    if float(deg) == 0:
        return FLAT_XML
    return XML_DIR / f"scene_slope{deg}_no_head.xml"


def write_courses():
    XML_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for deg in SLOPES:
        path = xml_path(deg)
        path.write_text(xml_text(deg), encoding='utf-8')
        paths.append(path)
    return paths


def ground_geom_ids(model):
    ids = []
    for name in GROUND_NAMES:
        try:
            ids.append(int(model.geom(name).id))
        except Exception:
            pass
    return np.asarray(ids, dtype=np.int32)


def native_model(deg=0):
    import mujoco
    path = xml_path(deg)
    old_cwd = os.getcwd()
    try:
        os.chdir(XML_DIR)
        return mujoco.MjModel.from_xml_path(path.name)
    finally:
        os.chdir(old_cwd)


def geom_map(current_model):
    """Map current geom ids to flat-scene ids.

    Generated slope XMLs prepend up_ramp/top_flat/down_ramp. The remaining
    geoms keep the flat model order. Extra ground geoms get -1 so randomization
    leaves their nominal contact properties in place.
    """
    base_n = native_model(0).ngeom
    n = int(current_model.ngeom)
    extra = n - base_n
    if extra < 0:
        raise ValueError('Current model has fewer geoms than flat baseline')
    return np.concatenate([np.full(extra, -1, dtype=np.int32),
                           np.arange(base_n, dtype=np.int32)])

def ground_geom_names(deg):
    return tuple(['floor'] + ([] if float(deg) == 0 else ['up_ramp', 'top_flat', 'down_ramp']))


def model_for_acceptance(deg, walls=True):
    import mujoco
    path = xml_path(deg)
    old_cwd = os.getcwd()
    try:
        os.chdir(XML_DIR)
        spec = mujoco.MjSpec.from_file(path.name)
    finally:
        os.chdir(old_cwd)
    if walls:
        for name, y in (("wall_left", -0.65), ("wall_right", 0.65)):
            g = spec.worldbody.add_geom()
            g.name = name
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [3.5, .05, .15]
            g.pos = [2.5, y, .15]
            g.contype = 1
            g.conaffinity = 1
            g.rgba = [.55, .55, .6, 1]
        g = spec.worldbody.add_geom()
        g.name = "finish_marker"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [.02, .55, .002]
        g.pos = [5.0, 0., .003]
        g.contype = 0
        g.conaffinity = 0
        g.rgba = [.1, .9, .2, 1]
    old_cwd = os.getcwd()
    try:
        os.chdir(XML_DIR)
        model = spec.compile()
    finally:
        os.chdir(old_cwd)
    model.opt.timestep = .002
    model.vis.global_.offwidth = 800
    model.vis.global_.offheight = 600
    return model
