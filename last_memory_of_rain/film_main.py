#!/usr/bin/env python3
"""
THE LAST MEMORY OF RAIN — an original animated short film.
Procedural Blender-Python production: build -> benchmark -> render -> soundtrack -> print/grade -> assemble.
Style target: CINEMATIC 2.5D PIXEL ART (see reference/CINEMATIC_2.5D_PIXEL_ART_VISUAL_PRODUCTION_BIBLE.md).

Runtime: 165.0 s exactly (1320 unique frames @ 8 fps animation grid, conformed
to 24 fps at assembly by frame triplication; limited/stop-motion cadence).
Final upscale: 640x270 -> 1920x810 (x3 nearest-neighbor). Output keeps 2.37:1.
"""
import sys, os, math, time, json, shutil, argparse
import numpy as np
import bpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ENV = HERE / "_env"
FRAMES_DIR = HERE / "frames"
POST_DIR = HERE / "frames_post"
BENCH_DIR = HERE / "frames_bench"
AUDIO_WAV = HERE / "audio.wav"
FINAL_MP4 = HERE / "THE_LAST_MEMORY_OF_RAIN_FINAL.mp4"
PREVIEW_MP4 = HERE / "THE_LAST_MEMORY_OF_RAIN_preview.mp4"
BLEND_OUT = HERE / "film_project.blend"
README_OUT = HERE / "README.txt"
RENDER_LOG = HERE / "render_log.txt"
STORYBOARD = HERE / "storyboard.md"

FPS_RENDER = 8                 # animation grid
FPS_FINAL = 24                 # conformed output (triplication)
RES_W, RES_H = 640, 270
FRAME_END = 1320               # 165.0 s @ 8 fps
DURATION_S = 165.0
AUDIO_RATE = 44100
AUDIO_CH = 2

PRESETS = {
    "A": {"w": 640, "h": 270, "samples": 6, "max_bounces": 2, "seed": 101},
    "B": {"w": 640, "h": 270, "samples": 5, "max_bounces": 2, "seed": 101},
    "C": {"w": 640, "h": 270, "samples": 4, "max_bounces": 1, "seed": 101},
}
PRESET_ORDER = ["A", "B", "C"]
BUDGET_MIN = 190
FORCE_PRESET = None

_t0 = time.time()
def log_line(msg):
    line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(RENDER_LOG, "a") as fh: fh.write(line + "\n")
    except OSError:
        pass

def log_section(title):
    bar = "=" * 78
    log_line(bar); log_line(title); log_line(bar)

def frame_path(d, n):
    return d / ("frame_%04d.png" % n)

def frame_is_valid(path):
    try:
        import struct
        with open(path, "rb") as fh:
            sig = fh.read(33)
        if len(sig) < 33 or sig[:8] != b"\x89PNG\r\n\x1a\n":
            return False
        w, h = struct.unpack(">II", sig[16:24])
        return (w, h) == (RES_W, RES_H) and path.stat().st_size > 1500
    except OSError:
        return False

def reset_scene():
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.lights, bpy.data.cameras,
                 bpy.data.materials, bpy.data.worlds, bpy.data.images):
        for item in list(coll):
            try:
                coll.remove(item)
            except Exception:
                pass
    sc = bpy.context.scene
    sc.frame_start = 1; sc.frame_end = FRAME_END; sc.frame_set(1)
    return sc

def flat(obj):
    for p in obj.data.polygons: p.use_smooth = False

def mat_flat(name, color, rough=0.9, metal=0.0, emissive=None, emit_strength=0.0,
             transmission=0.0, alpha=1.0, cull=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    if cull:
        m.use_backface_culling = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metal
    if emissive is not None:
        try:
            bsdf.inputs["Emission Color"].default_value = (*emissive, 1.0)
        except KeyError:
            bsdf.inputs["Emission"].default_value = (*emissive, 1.0)
        try:
            bsdf.inputs["Emission Strength"].default_value = emit_strength
        except KeyError:
            pass
    if transmission:
        try: bsdf.inputs["Transmission Weight"].default_value = transmission
        except KeyError: pass
    if alpha < 1.0:
        try: m.surface_render_method = 'BLENDED'
        except Exception:
            try: m.surface_render_method = 'DITHERED'
            except Exception: pass
        try: m.blend_method = 'BLEND'
        except Exception: pass
        bsdf.inputs["Alpha"].default_value = alpha
    return m

def mat_generated(name, img, emit_strength=0.0, emit_color=(1, 1, 1), rough=0.9, pure_emission=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = rough
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = 'Closest'
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*emit_color, 1.0)
    em.inputs["Strength"].default_value = emit_strength if pure_emission else max(emit_strength, 1.0)
    if pure_emission:
        bsdf.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], bsdf.inputs["Emission Color"] if "Emission Color" in bsdf.inputs else bsdf.inputs["Emission"])
        try: bsdf.inputs["Emission Strength"].default_value = 1.0
        except KeyError: pass
    elif emit_strength > 0.0:
        nt.links.new(tex.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs["Emission"], bsdf.inputs["Emission Color"] if "Emission Color" in bsdf.inputs else bsdf.inputs["Emission"])
        try: bsdf.inputs["Emission Strength"].default_value = 1.0
        except KeyError: pass
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return m

def add_uv_sphere(name, loc, scale, mat, segs=16, rings=8):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=rings, location=loc)
    o = bpy.context.active_object; o.name = name; o.scale = scale
    o.data.materials.append(mat); return o

def add_cylinder(name, loc, radius, depth, mat, verts=16, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name
    o.data.materials.append(mat); return o

def add_box(name, loc, scale, mat, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name; o.scale = scale
    o.data.materials.append(mat); return o

def add_cone(name, loc, r1, r2, depth, mat, verts=16, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r2, radius2=r1, depth=depth, location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name
    o.data.materials.append(mat); return o

def add_plane(name, loc, scale, mat, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name; o.scale = scale
    o.data.materials.append(mat); return o

def empty_at(name, loc):
    o = bpy.data.objects.new(name, None)
    o.location = loc
    bpy.context.collection.objects.link(o)
    return o

def add_text(name, body, loc, size, mat, rot=(math.radians(90), 0, 0), align='CENTER'):
    bpy.ops.object.text_add(location=loc, rotation=rot)
    o = bpy.context.active_object; o.name = name
    o.data.body = body; o.data.size = size; o.data.align_x = align
    o.data.materials.append(mat); return o

def canvas_image(name, w, h, draw_fn, flip_rows=True):
    px = np.zeros((h, w, 3), dtype=np.uint8)
    draw_fn(px)
    img = bpy.data.images.new(name, w, h, alpha=False)
    img.colorspace_settings.name = 'Non-Color'
    arr = np.zeros((h, w, 4), dtype=np.float32)
    arr[..., :3] = px.astype(np.float32) / 255.0
    arr[..., 3] = 1.0
    img.pixels = (arr[::-1] if flip_rows else arr).ravel().tolist()
    img.pack()
    return img

def world_sky_canvas(px):
    h, w, _ = px.shape
    zenith = np.array([8, 34, 44], np.float32)
    mid    = np.array([20, 58, 62], np.float32)
    rose   = np.array([112, 74, 60], np.float32)
    horizon= np.array([212, 146, 88], np.float32)
    ember  = np.array([118, 66, 36], np.float32)
    for y in range(h):
        t = y / (h - 1)
        if t < 0.44:
            a = t / 0.44; c = zenith * (1 - a) + mid * a
        elif t < 0.56:
            a = (t - 0.44) / 0.12; c = mid * (1 - a) + rose * a
        elif t < 0.62:
            a = (t - 0.56) / 0.06; c = rose * (1 - a) + horizon * a
        elif t < 0.68:
            a = (t - 0.62) / 0.06; c = horizon * (1 - a) + ember * a
        else:
            a = (t - 0.68) / 0.32; c = ember * (1 - a) + zenith * a
        px[y, :] = c.astype(np.uint8)

def sky_canvas(px):
    h, w, _ = px.shape
    zenith = np.array([8, 34, 44], np.float32)
    mid    = np.array([20, 58, 62], np.float32)
    rose   = np.array([112, 74, 60], np.float32)
    horizon= np.array([212, 146, 88], np.float32)
    ember  = np.array([118, 66, 36], np.float32)
    for y in range(h):
        t = y / (h - 1)
        if t < 0.34:
            a = t / 0.34; c = zenith * (1 - a) + mid * a
        elif t < 0.46:
            a = (t - 0.34) / 0.12; c = mid * (1 - a) + rose * a
        elif t < 0.56:
            a = (t - 0.46) / 0.10; c = rose * (1 - a) + horizon * a
        else:
            a = (t - 0.56) / 0.44; c = horizon * (1 - a) + ember * a
        px[y, :] = c.astype(np.uint8)

def crust_canvas(px):
    rng = np.random.default_rng(1414)
    h, w, _ = px.shape
    px[:] = (64, 51, 38)
    px[:] = px.astype(np.float32) + rng.normal(0, 14, (h, w, 1))
    px[:] = px.astype(np.uint8)
    for _ in range(90):
        x, y = rng.integers(0, w), rng.integers(0, h)
        ang = rng.random() * 2 * math.pi
        for _ in range(int(rng.integers(60, 260))):
            x += int(math.cos(ang) * 2 + rng.normal(0, 0.8)); y += int(math.sin(ang) * 2 + rng.normal(0, 0.8))
            ang += rng.normal(0, 0.22)
            if 0 <= x < w and 0 <= y < h:
                px[y, x] = (46, 36, 26)
                if rng.random() < 0.4:
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        xx, yy = x + dx, y + dy
                        if 0 <= xx < w and 0 <= yy < h: px[yy, xx] = (60, 48, 36)
    for _ in range(2600):
        x, y = rng.integers(0, w), rng.integers(0, h)
        px[y, x] = tuple(int(v) for v in (px[y, x].astype(np.int16) + rng.choice([-40, 35])).clip(20, 190))

def glyph_rain_canvas(px):
    rng = np.random.default_rng(777)
    h, w, _ = px.shape
    px[:] = (8, 14, 16)
    for _ in range(46):   # falling-dash rain motif
        x = int(rng.integers(6, w - 6)); y0 = int(rng.integers(0, h - 40)); ln = int(rng.integers(10, 34))
        v = rng.random()
        col = (120, 210, 205) if v < 0.75 else (235, 240, 220)
        for i in range(ln):
            y = y0 + i
            if 0 <= y < h: px[y, x] = col
    for _ in range(10):   # cloud glyph chevrons
        cx = int(rng.integers(20, w - 20)); cy = int(rng.integers(int(h * 0.55), h - 12))
        for k in range(14):
            px[cy - k // 2, cx - k] = (140, 220, 215); px[cy - k // 2, cx + k] = (140, 220, 215)

def build_world(sc):
    world = bpy.data.worlds.new("PRL_WORLD") if not sc.world else sc.world
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = WORLD_SKY_IMG[0]
    bg.inputs["Strength"].default_value = 0.75
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

def build_exterior(M):
    crust_img = canvas_image("CRUST_IMG", 512, 512, crust_canvas)
    crust_mat = mat_generated("CRUST_MT", crust_img, emit_strength=0.0, rough=0.97)
    add_plane("CRUST", (0, 0, 0.0), (170, 170, 1), crust_mat, rot=(0, 0, 0))
    # dunes far (east ridge) — stacked low cones, unlit-ish silhouette
    add_cone("DUNE_FAR_1", (-38.0, 52.0, 2.6), 8.0, 46.0, 5.2, M["dune_far"], verts=9)
    add_cone("DUNE_FAR_2", (18.0, 74.0, 3.4), 10.0, 62.0, 6.8, M["dune_far"], verts=9)
    add_cone("DUNE_FAR_3", (64.0, 44.0, 2.2), 7.0, 38.0, 4.4, M["dune_far"], verts=8)
    add_cone("DUNE_FAR_4", (-70.0, 30.0, 2.0), 6.0, 36.0, 4.0, M["dune_far"], verts=8)
    # dunes near
    add_cone("DUNE_NEAR_1", (-17.0, 21.0, 1.35), 5.0, 24.0, 2.7, M["dune_near"], verts=8)
    add_cone("DUNE_NEAR_2", (26.0, 24.0, 1.1), 4.0, 19.0, 2.2, M["dune_near"], verts=8)
    add_cone("DUNE_NEAR_3", (-3.0, 38.0, 1.6), 6.0, 26.0, 3.2, M["dune_near"], verts=8)
    # dead trees (east, near the pad road)
    for (tx, ty, s) in [(8.2, -6.0, 1.0), (10.6, -2.4, 0.72), (-7.8, -7.6, 0.85), (12.4, 3.2, 0.6)]:
        add_cylinder("DEADT_TRUNK", (tx, ty, 0.75 * s), 0.075 * s, 1.5 * s, M["deadwood"], verts=6)
        add_cylinder("DEADT_BR_1", (tx + 0.28 * s, ty, 1.25 * s), 0.035 * s, 0.9 * s, M["deadwood"], verts=5, rot=(0, math.radians(48), 0))
        add_cylinder("DEADT_BR_2", (tx - 0.24 * s, ty + 0.1 * s, 0.95 * s), 0.03 * s, 0.7 * s, M["deadwood"], verts=5, rot=(math.radians(20), math.radians(-52), 0))

def build_storm_cell(M):
    """Distant dark cell silhouetted on the copper horizon (+y). Flat dark shells."""
    cell = mat_flat("CELL_DARK", (0.030, 0.044, 0.056), 1.0, emissive=(0.030, 0.044, 0.056), emit_strength=0.55)
    add_uv_sphere("CELL_MAIN", (2.5, 84.0, 7.5), (17.0, 4.0, 6.2), cell, segs=18, rings=9)
    add_uv_sphere("CELL_TOWER", (11.5, 81.0, 10.5), (9.5, 3.4, 6.8), cell, segs=16, rings=8)
    add_uv_sphere("CELL_LEFT", (-9.0, 82.0, 8.0), (10.5, 3.6, 4.6), cell, segs=16, rings=8)
    add_uv_sphere("CELL_RIGHT", (23.0, 85.0, 5.6), (9.0, 3.0, 3.4), cell, segs=14, rings=7)

def build_flock(M):
    """A flock of distant birds crossing during S03-A (frames ~400-428)."""
    dark = M["m07_joint"]
    flock = empty_at("FLOCK", (0, 0, 0))
    for i in range(5):
        b = add_cone("BIRD_%d" % i, (0, 0, 0), 0.055, 0.012, 0.42, dark, verts=4, rot=(0, math.radians(90), 0))
        b.parent = flock
        b.location = (-3.2 + i * 1.6, (i % 2) * 1.2 - 0.6, math.sin(i * 1.7) * 0.8)
    key_loc(flock, [(400, (-20.0, 22.0, 10.2)), (428, (20.0, 22.0, 11.2))])

def build_station(M):
    """WS-M07 weather annex: annex box + concrete pedestal + dome mast."""
    st = empty_at("STATION", (0, 0, 0))
    add_box("PAD", (0, 0, 0.22), (11.5, 7.2, 0.45), M["padstone"])
    # concrete pedestal skirt (pad -> room floor)
    concrete = mat_flat("CONCRETE", (0.245, 0.235, 0.215), 0.96)
    for i, (px, py) in enumerate(((-2.7, -1.95), (2.7, -1.95), (-2.7, 1.95), (2.7, 1.95))):
        add_cone("PIER_%d" % i, (px, py, 2.0), 0.30, 0.46, 3.3, concrete, verts=4, rot=(0, 0, math.radians(45)))
    # dome house on a support mast above the box annex
    add_cylinder("DOME_MAST", (0, 0, 7.6), 0.30, 1.6, M["rust"], verts=10)
    dome = add_uv_sphere("DOME", (0, 0, 8.45), (1.6, 1.6, 0.95), M["patina_dark"], segs=18, rings=9)
    add_cylinder("PORTHOLE", (0, -2.76, 5.6), 0.34, 0.1, M["glass_dark"], verts=12, rot=(math.radians(90), 0, 0))
    # satellite dish on a roof mast (frame-right)
    add_cylinder("DISH_MAST", (2.1, 0.6, 8.45), 0.09, 2.3, M["rust"], verts=8)
    dish = add_cone("DISH", (2.1, 0.6, 9.75), 1.25, 0.18, 0.75, M["patina"], verts=18, rot=(math.radians(112), 0, math.radians(18)))
    # solar wing (frame-left, tilted)
    wing = add_box("SOLAR_WING", (-2.6, 0.9, 6.15), (3.3, 0.09, 1.15), M["solar"], rot=(math.radians(-24), 0, math.radians(14)))
    add_box("SOLAR_STRUT", (-1.5, 0.75, 5.6), (0.14, 0.14, 1.15), M["rust"], rot=(0, math.radians(24), 0))
    # anemometer / wind pump on far mast (left ridge)
    add_cylinder("PUMP_MAST", (-6.4, 3.2, 2.1), 0.12, 4.2, M["rust"], verts=8)
    rotor = empty_at("WIND_PUMP", (-6.4, 3.2, 4.35))
    for i in range(4):
        blade = add_box("PUMP_BLADE_%d" % i, (0, 0, 0), (0.06, 0.02, 1.05), M["rust_dark"])
        blade.parent = rotor
        blade.location = (math.sin(i * math.pi / 2) * 0.62, 0, math.cos(i * math.pi / 2) * 0.62)
        blade.rotation_euler = (math.radians(90), 0, i * math.pi / 2)
    # tattered wind sock on a short pole
    add_cylinder("SOCK_POLE", (3.4, -2.6, 1.35), 0.045, 2.7, M["rust"], verts=6)
    sock = add_cone("WIND_SOCK", (3.4, -2.95, 2.62), 0.16, 0.05, 0.85, M["sock_faded"], verts=10, rot=(math.radians(80), 0, 0))
    # antenna rods on the roof
    for i, ax in enumerate((-0.7, 0.0, 0.8)):
        add_cylinder("ANT_%d" % i, (ax, 0.2, 7.75 + 0.28 * i), 0.035, 1.0 + 0.35 * i, M["rust_dark"], verts=6)
    # rain-gauge stand (front pad)
    add_cylinder("GAUGE_STAND", (1.9, -2.9, 0.62), 0.05, 0.85, M["rust"], verts=6)
    add_cylinder("GAUGE_CAN", (1.9, -2.9, 1.1), 0.16, 0.34, M["patina"], verts=12)
    return st

def build_pad_props(M):
    # launch pad: blast ring + service structure
    add_cylinder("BLAST_RING", (4.6, 2.4, 0.55), 1.05, 0.35, M["scorch"], verts=18)
    add_cylinder("PAD_LUG", (3.35, 3.3, 0.28), 0.16, 0.55, M["rust"], verts=8)
    add_cylinder("PAD_LUG2", (5.85, 3.35, 0.28), 0.16, 0.55, M["rust"], verts=8)
    add_box("SERVICE_BOX", (6.1, 1.3, 0.75), (0.7, 0.55, 0.6), M["panel_dark"])
    add_cylinder("SERVICE_CAN", (6.1, 1.3, 1.38), 0.24, 0.5, M["lamp_warm"], verts=10)
    add_cylinder("FUEL_DRUM", (3.3, 0.9, 0.55), 0.34, 0.85, M["rust_dark"], verts=12)
    add_cylinder("FUEL_DRUM2", (2.6, 1.6, 0.42), 0.26, 0.6, M["patina"], verts=12)
    return

def build_materials():
    M = {}
    M["dune_far"]   = mat_flat("DUNE_FAR",  (0.155, 0.100, 0.072), 0.98)
    M["dune_near"]  = mat_flat("DUNE_NEAR", (0.205, 0.145, 0.098), 0.98)
    M["deadwood"]   = mat_flat("DEADWOOD",  (0.150, 0.105, 0.070), 0.95)
    M["padstone"]   = mat_flat("PADSTONE",  (0.325, 0.285, 0.235), 0.92)
    M["crust"]      = mat_flat("CRUST_FB",  (0.360, 0.285, 0.205), 0.97)
    M["rust"]       = mat_flat("RUST",      (0.360, 0.165, 0.075), 0.82, metal=0.25)
    M["rust_dark"]  = mat_flat("RUST_DARK", (0.205, 0.090, 0.045), 0.88, metal=0.2)
    M["patina"]     = mat_flat("PATINA",    (0.145, 0.315, 0.285), 0.72, metal=0.35, cull=True)
    M["patina_dark"]= mat_flat("PATINA_DK", (0.075, 0.205, 0.185), 0.78, metal=0.3, cull=True)
    M["glass_dark"] = mat_flat("GLASS_DK",  (0.045, 0.060, 0.062), 0.15, metal=0.1)
    M["solar"]      = mat_flat("SOLAR",     (0.065, 0.115, 0.175), 0.30, metal=0.75)
    M["sock_faded"] = mat_flat("SOCK_FD",   (0.475, 0.405, 0.275), 0.96)
    M["scorch"]     = mat_flat("SCORCH",    (0.065, 0.050, 0.042), 0.98)
    M["int_wall"]   = mat_flat("INT_WALL",  (0.255, 0.228, 0.190), 0.94)
    M["int_floor"]  = mat_flat("INT_FLOOR", (0.168, 0.145, 0.116), 0.92)
    M["panel_dark"] = mat_flat("PANEL_DK",  (0.105, 0.105, 0.112), 0.55, metal=0.35)
    M["brass"]      = mat_flat("BRASS",     (0.430, 0.330, 0.150), 0.38, metal=0.85)
    M["paper"]      = mat_flat("PAPER",     (0.600, 0.565, 0.465), 0.95)
    M["wood"]       = mat_flat("WOOD",      (0.255, 0.160, 0.085), 0.82)
    M["m07_body"]   = mat_flat("M07_BODY",  (0.515, 0.495, 0.445), 0.58, metal=0.25)
    M["m07_dark"]   = mat_flat("M07_DARK",  (0.260, 0.250, 0.235), 0.62, metal=0.3)
    M["m07_joint"]  = mat_flat("M07_JOINT", (0.150, 0.145, 0.135), 0.52, metal=0.4)
    M["scarf"]      = mat_flat("SCARF",     (0.715, 0.135, 0.088), 0.94)   # THE red
    M["velvet"]     = mat_flat("VELVET",    (0.185, 0.118, 0.115), 0.98)
    M["lens_cyan"]  = mat_flat("LENS",      (0.10, 0.35, 0.40), 0.15, metal=0.1, emissive=(0.30, 0.95, 1.15), emit_strength=0.0)
    M["seed_glass"] = mat_flat("SEEDGLASS", (0.82, 0.73, 0.46), 0.10, emissive=(1.0, 0.72, 0.28), emit_strength=0.30, transmission=0.72)
    M["seed_amber"] = mat_flat("SEED",      (0.85, 0.52, 0.16), 0.35, emissive=(1.0, 0.62, 0.18), emit_strength=0.9)
    M["core_rocket"]= mat_flat("CORE_ROCK", (0.62, 0.50, 0.26), 0.25, metal=0.6, emissive=(1.0, 0.82, 0.42), emit_strength=1.0)
    M["core_keeper"]= mat_flat("CORE_KEEP", (0.30, 0.50, 0.44), 0.25, metal=0.6, emissive=(0.45, 0.95, 0.85), emit_strength=0.9)
    M["lamp_warm"]  = mat_flat("LAMP_WARM", (0.9, 0.78, 0.55), 0.4, emissive=(1.0, 0.82, 0.52), emit_strength=1.5)
    M["strip_warm"] = mat_flat("STRIP",     (0.85, 0.72, 0.5), 0.5, emissive=(1.0, 0.83, 0.55), emit_strength=2.0)
    M["screen_off"] = mat_flat("SCREEN_OFF",(0.07, 0.075, 0.085), 0.35, metal=0.2)
    M["screen_cyan"]= mat_flat("SCREEN_CY", (0.06, 0.10, 0.11), 0.3, emissive=(0.25, 0.75, 0.85), emit_strength=0.0)
    M["green_photo"]= mat_flat("GREEN_PIC", (0.13, 0.34, 0.13), 0.85, emissive=(0.10, 0.32, 0.12), emit_strength=0.22)
    M["photo_frame"]= mat_flat("PHOTO_FR",  (0.16, 0.10, 0.05), 0.6)
    M["leaf"]       = mat_flat("LEAF",      (0.16, 0.46, 0.14), 0.75, emissive=(0.09, 0.30, 0.08), emit_strength=0.38)
    M["hilt_white"] = mat_flat("SPIRIT",    (0.92, 0.93, 0.88), 0.5, emissive=(1.0, 1.0, 0.95), emit_strength=1.1)
    return M

def build_interior(M):
    """The dome interior room, 6.9 x 5.3 x 3.4 m, floor z = 3.75."""
    FX, FY, FZ = 3.75, 0.0, 3.75
    add_box("INT_FLOOR2", (0, 0, FZ - 0.15), (6.9, 5.3, 0.3), M["int_floor"])
    add_box("INT_CEIL",   (0, 0, FZ + 3.55), (6.9, 5.3, 0.3), M["int_floor"])
    add_box("INT_WALL_B", (0, -2.62, FZ + 1.7), (6.9, 0.28, 3.4), M["int_wall"])
    add_box("INT_WALL_L", (-3.32, 0, FZ + 1.7), (0.28, 5.3, 3.4), M["int_wall"])
    add_box("INT_WALL_R", ( 3.32, 0, FZ + 1.7), (0.28, 5.3, 3.4), M["int_wall"])
    add_box("INT_WALL_F", (0,  2.62, FZ + 1.7), (6.9, 0.28, 3.4), M["int_wall"])
    # light slit (front-upper-left) — emissive blade
    slit = add_plane("LIGHT_SLIT", (-2.35, 2.46, FZ + 3.0), (1.5, 0.75, 1), M["lamp_warm"], rot=(0, 0, 0))
    # --- back wall: gauge panel ---
    add_box("GAUGE_PANEL", (0, -2.44, FZ + 1.55), (3.1, 0.14, 1.5), M["panel_dark"])
    for i, gx in enumerate((-1.0, 0.0, 1.0)):
        add_cylinder("GAUGE_%d" % i, (gx, -2.35, FZ + 1.78), 0.20, 0.10, M["brass"], verts=16, rot=(math.radians(90), 0, 0))
        add_cylinder("GAUGE_FACE_%d" % i, (gx, -2.30, FZ + 1.78), 0.155, 0.02, M["paper"], verts=16, rot=(math.radians(90), 0, 0))
        needle = add_box("NEEDLE_%d" % i, (gx, -2.27, FZ + 1.78), (0.016, 0.02, 0.125), M["m07_joint"], rot=(0, math.radians(38), math.radians(-52 + 30 * i + (137 if i == 2 else 0))))
    # rain gauge readout (eleven-year zero)
    add_text("TXT_ZERO", "0.0 mm", (0, -2.29, FZ + 1.02), 0.30, M["lamp_warm"], rot=(math.radians(90), 0, 0))
    add_text("TXT_DAYS", "11,324 consecutive days", (0, -2.29, FZ + 0.72), 0.105, M["lamp_warm"], rot=(math.radians(90), 0, 0))
    # barometer pendulum toy (animated)
    pend = empty_at("PENDULUM", (-2.45, -1.4, FZ + 2.15))
    add_cylinder("PEND_ARM", (0, 0, -0.42), 0.018, 0.85, M["brass"], verts=6).parent = pend
    add_cylinder("PEND_BOB", (0, 0, -0.86), 0.10, 0.05, M["brass"], verts=12, rot=(math.radians(90), 0, 0)).parent = pend
    # --- cradle (left) ---
    add_box("CRADLE_PLINTH", (-1.95, 0.55, FZ + 0.18), (1.25, 1.05, 0.36), M["panel_dark"])
    add_cylinder("CRADLE_RING", (-1.95, 0.55, FZ + 0.44), 0.52, 0.16, M["m07_dark"], verts=18)
    # --- drawer cabinet (right wall) ---
    add_box("CABINET", (2.45, -1.35, FZ + 0.55), (1.35, 0.85, 1.1), M["wood"])
    drawer = add_box("DRAWER", (2.45, -1.16, FZ + 0.72), (1.06, 0.52, 0.22), M["wood"])
    add_box("DRAWER_PULL", (2.45, -0.885, FZ + 0.72), (0.34, 0.05, 0.045), M["brass"]).parent = drawer
    # seed glass sits inside drawer
    glass = add_cylinder("SEED_GLASS", (2.45, -1.05, FZ + 0.93), 0.155, 0.42, M["seed_glass"], verts=12)
    seed = add_uv_sphere("SEED", (2.45, -1.05, FZ + 0.88), (0.075, 0.075, 0.10), M["seed_amber"], segs=10, rings=6)
    # framed photograph of green
    add_box("PHOTO_FRAME", (-1.55, -2.47, FZ + 2.35), (0.78, 0.07, 0.95), M["photo_frame"])
    add_plane("PHOTO_GREEN", (-1.55, -2.42, FZ + 2.35), (0.62, 0.79, 1), M["green_photo"], rot=(math.radians(90), 0, 0))
    # --- projector (upper right) + glyph plane on back wall ---
    add_box("PROJECTOR", (2.15, -0.7, FZ + 2.65), (0.62, 0.42, 0.32), M["panel_dark"])
    add_cylinder("PROJ_LENS", (2.15, -0.98, FZ + 2.62), 0.11, 0.12, M["brass"], verts=12, rot=(math.radians(90), 0, 0))
    glyph_img = canvas_image("GLYPH_RAIN", 384, 288, glyph_rain_canvas)
    glyph_mat = mat_generated("GLYPH_MAT", glyph_img, emit_strength=1.35, rough=0.7)
    M["GLYPH_MAT"] = glyph_mat
    add_plane("WALL_GLYPH", (0.35, -2.45, FZ + 1.35), (2.5, 1.85, 1), glyph_mat, rot=(math.radians(90), 0, 0))
    # --- terminal desk (right) + 61% glyph ---
    add_box("TERM_DESK", (2.15, 1.15, FZ + 0.55), (1.5, 0.8, 1.1), M["panel_dark"])
    add_box("TERM_SCREEN", (2.15, 1.28, FZ + 1.35), (0.95, 0.09, 0.62), M["screen_off"], rot=(math.radians(-18), 0, 0))
    pct = add_text("TXT_PCT", "61%", (2.15, 1.235, FZ + 1.40), 0.30, M["screen_cyan"], rot=(math.radians(90 - 18), 0, 0))
    pct.scale = (0.001, 0.001, 0.001)  # hidden until S03-B
    add_text("TXT_RAIN", "RAIN", (2.15, 1.235, FZ + 1.16), 0.11, M["screen_cyan"], rot=(math.radians(90 - 18), 0, 0))
    # --- workbench with the two cores (front-center) ---
    add_box("WORKBENCH", (0, 0.95, FZ + 0.42), (2.3, 0.95, 0.84), M["wood"])
    add_cylinder("CORE_ROCKET", (-0.55, 0.95, FZ + 0.98), 0.21, 0.55, M["core_rocket"], verts=14)
    add_cylinder("CORE_KEEPER", ( 0.55, 0.95, FZ + 0.98), 0.21, 0.55, M["core_keeper"], verts=14)
    add_box("CORE_BED", (0, 0.95, FZ + 0.88), (1.85, 0.68, 0.09), M["panel_dark"])
    ring1 = add_cone("HOLO_RING1", (0, 0.95, FZ + 1.75), 0.34, 0.34, 0.05, M["screen_cyan"], verts=24, rot=(math.radians(90), 0, 0))
    ring2 = add_cone("HOLO_RING2", (0, 0.95, FZ + 1.75), 0.22, 0.22, 0.05, M["screen_cyan"], verts=24, rot=(math.radians(55), 0, math.radians(20)))
    # ceiling light strips (power-down show)
    strip1 = add_box("STRIP1", (-1.5, 0, FZ + 3.36), (1.7, 0.16, 0.07), M["strip_warm"])
    strip2 = add_box("STRIP2", ( 1.5, 0, FZ + 3.36), (1.7, 0.16, 0.07), M["strip_warm"])
    return

def build_rocket(M):
    """The drought-relief cloud-seeder rocket on its pad."""
    root = empty_at("ROCKET", (4.6, 2.4, 1.15))
    body = add_cylinder("ROCKET_BODY", (0, 0, 1.15), 0.52, 2.3, M["padstone"], verts=16)
    body.parent = root; body.location = (0, 0, 1.15)
    nose = add_cone("ROCKET_NOSE", (0, 0, 2.9), 0.02, 0.5, 1.1, M["rust"], verts=16)
    nose.parent = root; nose.location = (0, 0, 2.9)
    finL = add_box("ROCKET_FIN_L", (0, 0, 0), (0.1, 0.7, 0.9), M["rust_dark"])
    finL.parent = root; finL.location = (-0.55, 0, 0.45); finL.rotation_euler = (0, math.radians(-14), 0)
    finR = add_box("ROCKET_FIN_R", (0, 0, 0), (0.1, 0.7, 0.9), M["rust_dark"])
    finR.parent = root; finR.location = (0.55, 0, 0.45); finR.rotation_euler = (0, math.radians(14), 0)
    finB = add_box("ROCKET_FIN_B", (0, 0, 0), (0.7, 0.1, 0.9), M["rust_dark"])
    finB.parent = root; finB.location = (0, -0.55, 0.45); finB.rotation_euler = (math.radians(-14), 0, 0)
    band = add_cylinder("ROCKET_BAND", (0, 0, 0), 0.53, 0.22, M["scorch"], verts=16)
    band.parent = root; band.location = (0, 0, 0.55)
    cyl = add_cylinder("ROCKET_CORE_PORT", (0, 0, 0), 0.16, 0.3, M["glass_dark"], verts=10)
    cyl.parent = root; cyl.location = (0, 0, 0.28)
    # engine flare (hidden until ignition)
    flare = add_cone("FLARE", (0, 0, 0), 0.36, 0.75, 1.35, M["lamp_warm"], verts=12, rot=(math.radians(180), 0, 0))
    flare.parent = root; flare.location = (0, 0, -0.75)
    return root

def build_m07(M):
    """M-07, the keeper. Proportions locked across all shots: ~1.9 m tall."""
    root = empty_at("M07", (-8.5, 19.0, 0.0))
    def part(name, builder, loc, rot=(0, 0, 0), parent=root):
        o = builder(name, loc, rot)
        o.parent = parent
        return o
    def p_box(n, loc, scale, mat, rot=(0, 0, 0)):
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc, rotation=rot)
        o = bpy.context.active_object; o.name = n; o.scale = scale
        o.data.materials.append(mat); return o
    def p_cyl(n, loc, r, d, mat, verts=12, rot=(0, 0, 0)):
        bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=d, location=loc, rotation=rot)
        o = bpy.context.active_object; o.name = n
        o.data.materials.append(mat); return o
    def p_sph(n, loc, scale, mat, segs=14, rings=8):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=rings, location=loc)
        o = bpy.context.active_object; o.name = n; o.scale = scale
        o.data.materials.append(mat); return o
    # legs (thigh + shin + foot); thighs are the animated swing pair
    for i, sx in ((0, -1), (1, 1)):
        th = p_cyl("M07_LEG_%d" % i, (sx * 0.17, 0, 0.62), 0.085, 0.42, M["m07_dark"], verts=10)
        sh = p_cyl("M07_SHIN_%d" % i, (sx * 0.17, 0, 0.19), 0.068, 0.30, M["m07_body"], verts=10)
        ft = p_box("M07_FOOT_%d" % i, (sx * 0.17, -0.035, 0.045), (0.19, 0.26, 0.09), M["m07_joint"])
        th.parent = root; sh.parent = root; ft.parent = root
    # pelvis + torso
    p = p_box("M07_PELVIS", (0, 0, 0.88), (0.42, 0.30, 0.20), M["m07_dark"]); p.parent = root
    t = p_box("M07_TORSO", (0, 0, 1.24), (0.56, 0.36, 0.52), M["m07_body"], rot=(math.radians(-4), 0, 0)); t.parent = root
    pl = p_box("M07_CHEST_PLATE", (0, -0.185, 1.24), (0.32, 0.045, 0.30), M["m07_dark"]); pl.parent = root
    # satchel (keeper's bag)
    sc = p_box("M07_SATCHEL", (0.06, 0.245, 1.18), (0.30, 0.16, 0.36), M["rust_dark"]); sc.parent = root
    st = p_box("M07_SATCHEL_FLAP", (0.06, 0.255, 1.34), (0.32, 0.17, 0.07), M["rust"]); st.parent = root
    # arms (shoulder dome + upper + fore + gripper); upper cylinders are the swing pair
    for i, sx in ((0, -1), (1, 1)):
        shp = p_sph("M07_SHOULDER_%d" % i, (sx * 0.36, 0, 1.40), (0.115, 0.115, 0.115), M["m07_dark"]); shp.parent = root
        up = p_cyl("M07_SH_%d" % i, (sx * 0.375, 0, 1.15), 0.055, 0.34, M["m07_body"], verts=8, rot=(0, math.radians(sx * 7), 0)); up.parent = root
        fo = p_cyl("M07_FORE_%d" % i, (sx * 0.40, 0, 0.86), 0.046, 0.32, M["m07_dark"], verts=8); fo.parent = root
        gr = add_cone("M07_GRIP_%d" % i, (sx * 0.41, 0, 0.63), 0.065, 0.02, 0.13, M["m07_joint"], verts=6, rot=(math.radians(180), 0, 0))
        gr.parent = root
    # head: flattened dome + faceplate + THE LENS + antenna
    hd = p_sph("M07_HEAD", (0, 0, 1.70), (0.285, 0.26, 0.23), M["m07_body"]); hd.parent = root
    fp = p_cyl("M07_FACEPLATE", (0, -0.155, 1.66), 0.16, 0.075, M["m07_dark"], verts=12, rot=(math.radians(90), 0, 0)); fp.parent = root
    ln = p_cyl("M07_LENS", (0, -0.20, 1.68), 0.075, 0.055, M["lens_cyan"], verts=12, rot=(math.radians(90), 0, 0)); ln.parent = root
    ant = p_cyl("M07_ANTENNA", (0.095, 0.03, 2.02), 0.016, 0.24, M["m07_joint"], verts=5); ant.parent = root
    tip = p_sph("M07_ANT_TIP", (0.095, 0.03, 2.15), (0.028, 0.028, 0.028), M["m07_joint"], segs=8, rings=5); tip.parent = root
    # THE SCARF (red — reserved): ring at the neck + tail down his right side
    bpy.ops.mesh.primitive_torus_add(major_radius=0.185, minor_radius=0.055, major_segments=14, minor_segments=8,
                                     location=(0, 0, 1.525), rotation=(math.radians(90), 0, 0))
    sf = bpy.context.active_object; sf.name = "SCARF_RING"; sf.data.materials.append(M["scarf"]); sf.parent = root
    tl = p_box("SCARF_TAIL", (0.175, 0.14, 1.24), (0.16, 0.055, 0.44), M["scarf"], rot=(math.radians(6), 0, math.radians(24))); tl.parent = root
    return root

def build_drop_rig(M):
    """S06-A macro rig: a big raindrop, a blade of grass, a splash ring."""
    blade = add_cone("DROP_BLADE", (0.9, -3.1, 0.32), 0.09, 0.012, 0.95, M["leaf"], verts=7, rot=(0, math.radians(11), math.radians(14)))
    drop = add_uv_sphere("DROP", (0.82, -3.02, 2.35), (0.085, 0.085, 0.105), M["glass_dark"], segs=12, rings=8)
    splash = add_cone("SPLASH", (0.79, -2.99, 0.335), 0.02, 0.42, 0.10, M["leaf"], verts=14)
    return blade, drop, splash

def build_sprout(M):
    """S06-C reward: the sprout in the seed glass (staging copy on the plinth)."""
    plinth = add_cylinder("SPROUT_STONE", (6.35, 1.15, 0.30), 0.42, 0.6, M["padstone"], verts=14)
    glass = add_cylinder("SPROUT_GLASS", (6.35, 1.15, 0.78), 0.155, 0.42, M["seed_glass"], verts=12)
    stem = add_cylinder("SPROUT_STEM", (6.35, 1.15, 1.06), 0.017, 0.16, M["leaf"], verts=6)
    leaf1 = add_uv_sphere("SPROUT_LEAF1", (6.28, 1.15, 1.15), (0.075, 0.028, 0.035), M["leaf"], segs=8, rings=4)
    leaf2 = add_uv_sphere("SPROUT_LEAF2", (6.42, 1.15, 1.155), (0.075, 0.028, 0.035), M["leaf"], segs=8, rings=4)
    leaf1.rotation_euler = (0, math.radians(28), math.radians(24))
    leaf2.rotation_euler = (0, math.radians(-28), math.radians(-24))
    return stem, leaf1, leaf2

def build_rain(M):
    """Particle rain visible in S06 (emitter over the pad/valley edge)."""
    emitter = add_plane("RAIN_EMITTER", (2.0, -1.0, 11.5), (34, 26, 1), M["int_floor"], rot=(math.radians(90), 0, 0))
    emitter.hide_render = True
    emitter.name = "RAIN_EMITTER"
    mod = emitter.modifiers.new("RAIN", 'PARTICLE_SYSTEM')
    ps = mod.particle_system
    ps.name = "RAIN_PS"
    settings = ps.settings
    settings.count = 650
    settings.frame_start = 881 - 30   # pre-roll so rain is already falling at the cut
    settings.frame_end = FRAME_END
    settings.lifetime = 24
    settings.emit_from = 'FACE'
    settings.physics_type = 'NEWTON'
    settings.particle_size = 0.055
    settings.normal_factor = 0.0
    settings.factor_random = 0.15
    settings.mass = 0.001
    settings.effector_weights.gravity = 1.0
    settings.render_type = 'OBJECT'
    streak = add_box("RAIN_STREAK", (0, 0, -50), (0.010, 0.010, 0.16), M["hilt_white"])
    streak.hide_render = False
    settings.instance_object = streak
    settings.use_rotations = False
    return emitter

def build_lights(sc):
    def sun(strength, color):
        o = bpy.data.objects.new("SUN", bpy.data.lights.new("SUN", 'SUN'))
        o.data.energy = strength; o.data.color = color
        o.rotation_euler = (math.radians(80), math.radians(8), math.radians(240))
        bpy.context.collection.objects.link(o); return o
    def area(name, loc, rot, strength, color, size):
        o = bpy.data.objects.new(name, bpy.data.lights.new(name, 'AREA'))
        o.data.energy = strength; o.data.color = color; o.data.size = size
        o.location = loc; o.rotation_euler = rot
        bpy.context.collection.objects.link(o); return o
    def point(name, loc, strength, color, size=0.35, shadow=True):
        o = bpy.data.objects.new(name, bpy.data.lights.new(name, 'POINT'))
        o.data.energy = strength; o.data.color = color; o.data.shadow_soft_size = size
        o.location = loc
        bpy.context.collection.objects.link(o); return o
    sunL = sun(1.15, (1.0, 0.70, 0.45))
    area("SKY_FILL", (0, 0, 30), (0, 0, 0), 0.22, (0.45, 0.62, 0.66), 40.0)
    p_cradle = point("P_CRADLE", (-1.95, 0.05, 5.65), 0.0, (1.0, 0.78, 0.52), 0.5)
    p_seed   = point("P_SEED",   ( 2.45, -1.0, 4.95), 0.0, (1.0, 0.66, 0.22), 0.28)
    p_term   = point("P_TERM",   ( 2.15,  1.3, 5.7),  0.0, (0.35, 0.85, 0.95), 0.4)
    p_core   = point("P_CORE",   ( 0.0,   0.95, 5.9), 0.0, (1.0, 0.78, 0.38), 0.6)
    p_flash  = point("P_FLASH",  ( 4.6,   2.4,  1.6),  0.0, (1.0, 0.92, 0.75), 1.2)
    p_dawn   = point("P_DAWN",   ( 6.3,   1.1,  1.6),  0.0, (1.0, 0.85, 0.66), 0.7)
    p_macro  = point("P_MACRO",  ( 1.5,  -3.7,  1.5),  0.0, (0.82, 0.90, 1.0), 0.45)
    p_fill   = point("P_FILL",   ( 0.0,   0.5,  4.6), 150.0, (0.62, 0.72, 0.80), 0.6)
    p_engine = point("P_ENGINE", ( 4.6,   2.4,  0.6),  0.0, (1.00, 0.62, 0.30), 0.7)
    return {"sun": sunL, "p_cradle": p_cradle, "p_seed": p_seed, "p_term": p_term,
            "p_core": p_core, "p_flash": p_flash, "p_dawn": p_dawn, "p_macro": p_macro,
            "p_fill": p_fill, "p_engine": p_engine}

def key_energy(light, keys):
    for f, v in keys:
        light.data.energy = v
        light.data.keyframe_insert("energy", frame=f)

def build_camera(sc):
    cam_data = bpy.data.cameras.new("CAM")
    cam_data.sensor_width = 36
    cam_data.clip_start = 0.05; cam_data.clip_end = 500.0
    cam = bpy.data.objects.new("CAM", cam_data)
    cam.location = (-13.5, 26.5, 2.6)
    bpy.context.collection.objects.link(cam)
    sc.camera = cam
    look = empty_at("CAM_LOOK", (0, 0, 5.6))
    con = cam.constraints.new('TRACK_TO')
    con.target = look
    con.track_axis = 'TRACK_NEGATIVE_Z'
    con.up_axis = 'UP_Y'
    keys = [
        (1,    (-13.5, 26.5, 2.6),  (0, 0, 5.6),  38),
        (72,   (-9.0, 21.5, 3.4),   (0, 0.5, 5.8),40),
        (73,   (-7.5, 17.5, 4.6),   (0, 0, 6.4),  46),
        (128,  (-3.2, 11.5, 5.8),   (0.4, 0, 6.6),54),
        (129,  (-0.6, 2.3, 5.45),   (-1.75, 0.2, 4.45), 40),
        (240,  (-1.15, 1.9, 5.1),   (-1.85, 0.3, 4.5), 46),
        (241,  (1.75, 0.55, 5.2),   (0.6, -2.1, 4.6), 44),
        (290,  (1.9, 1.05, 5.35),   (2.45, -0.55, 4.45), 42),
        (336,  (2.15, 0.9, 5.15),   (2.45, -0.75, 4.45), 44),
        (337,  (2.7, 1.65, 5.35),   (0.3, -2.45, 4.95), 40),
        (392,  (2.35, 1.3, 5.25),   (0.3, -2.45, 5.0), 44),
        (393,  (7.4, -3.6, 2.1),    (3.0, 40.0, 5.2), 55),
        (456,  (6.4, -4.0, 2.6),    (2.0, 40.0, 5.6), 58),
        (457,  (2.1, 2.35, 5.15),   (2.15, 1.25, 4.95), 38),
        (536,  (1.85, 1.95, 5.05),  (2.15, 1.27, 5.0), 46),
        (537,  (2.30, 2.20, 6.55),  (-0.35, 0.30, 4.60), 34),
        (624,  (1.85, 1.75, 6.00),  (-0.15, 0.55, 4.70), 37),
        (625,  (1.9, 1.85, 5.0),    (-0.6, 0.3, 4.8), 44),
        (704,  (1.35, 1.35, 5.0),   (-1.9, 0.45, 4.55), 48),
        (705,  (-2.6, -2.75, 1.95), (4.6, 2.4, 1.35), 42),
        (800,  (-1.15, -2.15, 1.75),(4.6, 2.4, 1.75), 48),
        (801,  (0.9, -2.35, 1.0),   (4.0, 2.2, 2.7), 47),
        (880,  (0.6, -1.9, 0.9),    (4.7, 2.6, 8.5), 55),
        (881,  (1.55, -4.3, 0.62),  (0.85, -2.99, 0.62), 68),
        (968,  (1.45, -4.12, 0.74), (0.83, -2.98, 0.60), 73),
        (969,  (-6.2, -8.5, 3.3),   (3.5, 1.0, 1.2), 36),
        (1076, (-4.6, -6.6, 2.9),   (4.6, 1.6, 0.9), 40),
        (1077, (4.75, -1.35, 1.95), (6.35, 1.15, 1.02), 38),
        (1180, (4.95, -1.10, 1.80), (6.35, 1.15, 1.10), 40),
        (1181, (2.3, -3.3, 1.5),    (5.5, 1.8, 1.15), 46),
        (1320, (1.6, -2.45, 1.35),  (5.5, 1.9, 1.35), 50),
    ]
    for f, pos, tgt, lens in keys:
        cam.location = pos
        cam.keyframe_insert("location", frame=f)
        look.location = tgt
        look.keyframe_insert("location", frame=f)
        cam.data.lens = lens
        cam.data.keyframe_insert("lens", frame=f)
    return cam

def kf(obj, attr, keys, fc=None):
    for f, v in keys:
        setattr(obj, attr, v)
        obj.keyframe_insert(attr, frame=f)

def key_loc(obj, keys):
    for f, v in keys:
        obj.location = v
        obj.keyframe_insert("location", frame=f)

def key_rot(obj, keys):
    for f, v in keys:
        obj.rotation_euler = v
        obj.keyframe_insert("rotation_euler", frame=f)

def key_scale(obj, keys):
    for f, v in keys:
        obj.scale = v
        obj.keyframe_insert("scale", frame=f)

def key_mat_emission(mat, keys):
    for f, v in keys:
        try:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            inp = bsdf.inputs.get("Emission Strength")
            if inp is not None:
                inp.default_value = v
                bsdf.inputs["Emission Strength"].keyframe_insert("default_value", frame=f)
                continue
        except Exception:
            pass
        try:
            em = mat.node_tree.nodes.get("Emission")
            if em is not None:
                em.inputs["Strength"].default_value = v
                em.inputs["Strength"].keyframe_insert("default_value", frame=f)
        except Exception:
            pass

def sin_keys(start, end, step, fn):
    return [(f, fn(f)) for f in range(start, end + 1, step)]

def animate_timeline(sc, L, M):
    O = bpy.data.objects
    m07 = O["M07"]
    # ---------- sun (dawn arc + lightning spikes) ----------
    key_energy(L["sun"], [(1, 1.15), (393, 1.0), (625, 0.55), (881, 0.4), (905, 10.0), (906, 0.4),
                          (927, 7.0), (928, 0.4), (1029, 0.4), (1030, 6.5), (1031, 0.4),
                          (1077, 2.0), (1181, 2.15), (1320, 2.2)])

    # ---------- interior practicals: power-on, power-down, dawn ----------
    key_energy(L["p_cradle"], [(128, 0.0), (133, 95.0), (625, 95.0), (650, 55.0), (672, 32.0),
                               (695, 14.0), (720, 5.0), (740, 0.0), (880, 0.0)])
    key_energy(L["p_seed"],   [(241, 0.0), (252, 36.0), (625, 36.0), (650, 17.0), (672, 8.0), (695, 0.0)])
    key_energy(L["p_term"],   [(457, 0.0), (459, 14.0), (462, 62.0), (705, 62.0), (730, 36.0), (760, 18.0), (800, 4.0), (815, 0.0)])
    key_energy(L["p_core"],   [(537, 0.0), (543, 22.0), (560, 34.0), (610, 20.0), (660, 27.0), (704, 22.0), (760, 15.0), (800, 9.0), (815, 0.0)])
    key_energy(L["p_flash"],  [(798, 0.0), (800, 320.0), (804, 90.0), (812, 18.0), (820, 0.0)])
    key_energy(L["p_dawn"],   [(1077, 0.0), (1120, 95.0), (1320, 105.0)])
    key_energy(L["p_engine"], [(795, 0.0), (800, 210.0), (870, 210.0), (878, 150.0), (881, 0.0)])
    # rocket engine light rides the rocket up (same loc keys as ROCKET with slight offset)
    key_loc(L["p_engine"],   [(800, (4.6, 2.4, 0.7)), (806, (4.6, 2.4, 0.75)), (815, (4.6, 2.4, 1.33)),
                              (825, (4.6, 2.4, 2.65)), (835, (4.6, 2.4, 4.95)), (845, (4.6, 2.4, 8.15)),
                              (855, (4.6, 2.4, 11.95)), (865, (4.6, 2.4, 16.15)), (875, (4.6, 2.4, 20.35)), (880, (4.6, 2.4, 24.0))])
    key_energy(L["p_macro"],  [(870, 0.0), (881, 24.0), (936, 32.0), (968, 27.0), (975, 0.0)])

    # ---------- material emission animation ----------
    strip_m = M["strip_warm"]; pct_m = M["screen_cyan"]; lens_m = M["lens_cyan"]; glyph_m = M["GLYPH_MAT"]
    key_mat_emission(strip_m, [(625, 2.0), (650, 1.1), (672, 0.55), (695, 0.2), (720, 0.05), (745, 0.0)])
    key_mat_emission(pct_m,   [(455, 0.0), (460, 1.6), (462, 0.4), (465, 1.9), (705, 1.9), (740, 1.1), (770, 0.4), (795, 0.0)])
    key_mat_emission(lens_m,  [(128, 0.0), (134, 0.6), (136, 3.0), (848, 3.0), (856, 2.0), (864, 1.1),
                               (872, 0.5), (880, 0.0), (1254, 0.0), (1256, 0.35), (1257, 3.3), (1320, 3.3)])
    key_mat_emission(glyph_m, [(337, 1.2), (339, 1.7), (341, 1.0), (343, 1.6), (346, 1.25), (349, 1.75),
                               (352, 1.1), (356, 1.5), (361, 1.3), (366, 1.65), (372, 1.15), (378, 1.45),
                               (384, 1.05), (392, 0.9)])

    # ---------- M-07: placement across shots ----------
    key_loc(m07, [(1, (-8.5, 19.0, 0.0)), (72, (-4.6, 12.6, 0.0)), (129, (-1.95, 0.55, 3.93)),
                  (240, (-1.95, 0.55, 3.93)), (241, (-1.95, 0.55, 3.93)), (256, (2.2, -0.62, 3.93)),
                  (290, (2.2, -0.62, 3.93)), (337, (0.3, 0.0, 3.93)), (392, (0.3, 0.0, 3.93)),
                  (393, (0.3, 0.0, 3.93)), (457, (2.15, 0.45, 3.93)), (536, (2.15, 0.45, 3.93)),
                  (537, (0.15, 1.8, 3.93)), (625, (0.15, 1.8, 3.93)), (704, (0.15, 1.8, 3.93)),
                  (705, (3.65, 1.35, 0.0)), (801, (3.65, 1.35, 0.0)),
                  (969, (4.35, 0.15, 0.0)), (1181, (4.35, 0.15, 0.0))])
    # walk bob + swing (S01-A)
    for f, z in sin_keys(1, 72, 6, lambda f: 0.02 * abs(math.sin(f * math.pi / 7.5))):
        m07.location.z = z; m07.keyframe_insert("location", frame=f)
    # legs / arms counter-swing
    legL, legR = O["M07_LEG_0"], O["M07_LEG_1"]
    armL, armR = O["M07_SH_0"], O["M07_SH_1"]
    for f in range(1, 73, 6):
        ph = f * math.pi / 7.5
        legL.rotation_euler[0] = 0.38 * math.sin(ph); legL.keyframe_insert("rotation_euler", index=0, frame=f)
        legR.rotation_euler[0] = -0.38 * math.sin(ph); legR.keyframe_insert("rotation_euler", index=0, frame=f)
        armL.rotation_euler[0] = -0.30 * math.sin(ph); armL.keyframe_insert("rotation_euler", index=0, frame=f)
        armR.rotation_euler[0] = 0.30 * math.sin(ph); armR.keyframe_insert("rotation_euler", index=0, frame=f)
    # reach for the drawer (right arm extends)
    key_rot(armR, [(241, (0, 0, 0)), (252, (math.radians(-95), 0, 0)), (262, (math.radians(-95), 0, 0)),
                   (272, (0, 0, 0))])
    # head: watch drawer (S02-B), watch cores (S04), watch rain (S06)
    head = O["M07_HEAD"]
    key_rot(head, [(241, (0, 0, 0)), (252, (math.radians(12), 0, math.radians(-28))),
                   (290, (math.radians(12), 0, math.radians(-28))), (337, (0, 0, 0)),
                   (457, (math.radians(8), 0, math.radians(18))), (536, (math.radians(8), 0, math.radians(18))),
                   (537, (0, 0, 0)), (969, (math.radians(-10), 0, 0)), (1076, (math.radians(-10), 0, 0)),
                   (1181, (0, 0, 0))])
    # collapse onto one knee (S05 -> S06): slump via leg + torso-free rotation keys on root parts
    for f, ang in [(852, 0.0), (862, 0.28), (880, 0.42), (969, 0.42)]:
        legL.rotation_euler[0] = ang; legL.keyframe_insert("rotation_euler", index=0, frame=f)
    m07.rotation_euler[0] = 0.0; m07.keyframe_insert("rotation_euler", index=0, frame=705)
    m07.rotation_euler[0] = 0.16; m07.keyframe_insert("rotation_euler", index=0, frame=880)
    m07.rotation_euler[0] = 0.16; m07.keyframe_insert("rotation_euler", index=0, frame=1076)
    m07.rotation_euler[0] = 0.0; m07.keyframe_insert("rotation_euler", index=0, frame=1181)

    # ---------- pendulum (interior life) ----------
    pend = O["PENDULUM"]
    for f in range(129, 720, 8):
        decay = max(0.0, 1.0 - max(0, f - 480) / 240.0)
        pend.rotation_euler[2] = math.radians(16.0) * math.sin(f * 0.22) * decay
        pend.keyframe_insert("rotation_euler", index=2, frame=f)

    # ---------- drawer slides open with the seed glass + seed riding along ----------
    drawer, glass, seed = O["DRAWER"], O["SEED_GLASS"], O["SEED"]
    key_loc(drawer, [(241, (2.45, -1.16, 4.47)), (252, (2.45, -1.16, 4.47)), (262, (2.45, -0.60, 4.47))])
    key_loc(glass,  [(241, (2.45, -1.05, 4.68)), (252, (2.45, -1.05, 4.68)), (262, (2.45, -0.49, 4.68))])
    key_loc(seed,   [(241, (2.45, -1.05, 4.63)), (252, (2.45, -1.05, 4.63)), (262, (2.45, -0.49, 4.63))])

    # ---------- holo rings (S04-A) ----------
    ring1, ring2 = O["HOLO_RING1"], O["HOLO_RING2"]
    for f in range(537, 705, 10):
        ring1.rotation_euler[2] = f * 0.09; ring1.keyframe_insert("rotation_euler", index=2, frame=f)
        ring2.rotation_euler[0] = f * 0.07; ring2.keyframe_insert("rotation_euler", index=0, frame=f)
    key_scale(ring1, [(648, (1, 1, 1)), (660, (0.001, 0.001, 0.001))])
    key_scale(ring2, [(648, (1, 1, 1)), (660, (0.001, 0.001, 0.001))])
    # 61% glyph scales in (S03-B)
    pct, rain_txt = O["TXT_PCT"], O["TXT_RAIN"]
    key_scale(pct, [(457, (0.001, 0.001, 0.001)), (461, (1.4, 1.4, 1.4)), (465, (1.0, 1.0, 1.0))])
    key_scale(rain_txt, [(457, (0.001, 0.001, 0.001)), (461, (1.3, 1.3, 1.3)), (465, (1.0, 1.0, 1.0))])

    # ---------- wind pump + sock ----------
    pump, sock = O["WIND_PUMP"], O["WIND_SOCK"]
    for i in range(24):
        pump.rotation_euler[1] = i * 1.9
        pump.keyframe_insert("rotation_euler", index=1, frame=1 + i * 54)
    for f, ang in [(969, 0.0), (1076, 9.0), (1180, 15.0), (1320, 23.0)]:
        pump.rotation_euler[1] = ang
        pump.keyframe_insert("rotation_euler", index=1, frame=f)
    for f in range(1, 1321, 30):   # sock dead-still sway
        sock.rotation_euler[0] = math.radians(80) + math.radians(1.2) * math.sin(f * 0.06)
        sock.keyframe_insert("rotation_euler", index=0, frame=f)
    sock.rotation_euler[0] = math.radians(58); sock.keyframe_insert("rotation_euler", index=0, frame=457)   # lifts, once
    sock.rotation_euler[0] = math.radians(80); sock.keyframe_insert("rotation_euler", index=0, frame=520)

    # ---------- rocket launch ----------
    rocket, flare = O["ROCKET"], O["FLARE"]
    key_loc(rocket, [(800, (4.6, 2.4, 1.15)), (805, (4.6, 2.4, 1.2)), (815, (4.6, 2.4, 1.78)),
                     (825, (4.6, 2.4, 3.1)), (835, (4.6, 2.4, 5.4)), (845, (4.6, 2.4, 8.6)),
                     (855, (4.6, 2.4, 12.4)), (865, (4.6, 2.4, 16.6)), (875, (4.6, 2.4, 20.8)), (880, (4.6, 2.4, 24.5))])
    for f in range(800, 831, 4):
        rocket.rotation_euler[2] = 0.02 * math.sin(f * 1.7)
        rocket.keyframe_insert("rotation_euler", index=2, frame=f)
    key_scale(flare, [(798, (0.001, 0.001, 0.001)), (800, (1.0, 1.0, 1.0)), (810, (1.15, 1.15, 1.3)),
                      (818, (0.9, 0.9, 1.0)), (824, (0.001, 0.001, 0.001))])

    # ---------- S06-A macro: the first drop ----------
    drop, blade, splash = O["DROP"], O["DROP_BLADE"], O["SPLASH"]
    key_scale(drop, [(881, (0.001, 0.001, 0.001)), (929, (1, 1, 1)), (935, (1, 1, 1)), (936, (0.001, 0.001, 0.001))])
    key_loc(drop, [(929, (0.82, -3.02, 2.3)), (931, (0.82, -3.02, 2.22)), (933, (0.82, -3.02, 2.0)),
                   (934, (0.82, -3.02, 1.45)), (935, (0.80, -2.99, 0.62))])
    key_rot(blade, [(929, (0, 0, 0)), (934, (0.10, 0.06, 0.02)), (940, (-0.08, -0.04, 0)), (946, (0.05, 0.02, 0)), (955, (0, 0, 0))])
    key_scale(splash, [(934, (0.001, 0.001, 0.001)), (936, (1.0, 1.0, 1.0)), (941, (1.6, 1.6, 1.0)), (947, (0.001, 0.001, 0.001))])

    # ---------- S06-C sprout ----------
    stem, leaf1, leaf2 = O["SPROUT_STEM"], O["SPROUT_LEAF1"], O["SPROUT_LEAF2"]
    key_scale(stem,  [(1000, (0.001, 0.001, 0.001)), (1085, (1, 1, 1)), (1180, (1, 1, 1))])
    key_scale(leaf1, [(1030, (0.001, 0.001, 0.001)), (1120, (0.28, 0.105, 0.13)), (1180, (0.28, 0.105, 0.13))])
    key_scale(leaf2, [(1035, (0.001, 0.001, 0.001)), (1125, (0.28, 0.105, 0.13)), (1180, (0.28, 0.105, 0.13))])

# ============================================================================
# SCENE ORCHESTRATION
# ============================================================================
WORLD_SKY_IMG = []

def build_scene():
    log_section("BUILDING PRODUCTION SCENE")
    sc = reset_scene()
    world_img = canvas_image("WORLD_SKY", 640, 360, world_sky_canvas, flip_rows=False)
    WORLD_SKY_IMG.clear(); WORLD_SKY_IMG.append(world_img)
    build_world(sc)
    M = build_materials()
    build_exterior(M)
    build_storm_cell(M)
    build_flock(M)
    build_station(M)
    build_pad_props(M)
    build_interior(M)
    build_rocket(M)
    build_m07(M)
    build_drop_rig(M)
    build_sprout(M)
    build_rain(M)
    L = build_lights(sc)
    cam = build_camera(sc)
    animate_timeline(sc, L, M)
    INTERIOR_BOOST = 6.0
    BOOST_SKIP = ("P_ENGINE", "P_FILL")
    for li in bpy.data.lights:
        if li.type != 'POINT' or li.name in BOOST_SKIP:
            continue
        li.energy *= INTERIOR_BOOST
    for ob in bpy.data.objects:
        if ob.type != 'LIGHT' or ob.data.type != 'POINT' or ob.data.name in BOOST_SKIP:
            continue
        ad = ob.animation_data
        if not ad or not ad.action:
            continue
        for fcu in ad.action.fcurves:
            if fcu.data_path == 'energy':
                for kp in fcu.keyframe_points:
                    kp.co.y *= INTERIOR_BOOST
                    kp.handle_left.y *= INTERIOR_BOOST
                    kp.handle_right.y *= INTERIOR_BOOST
                fcu.update()
    for ob in sc.objects:
        if ob.type == 'MESH':
            flat(ob)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT), check_existing=False)
    log_line("Scene build complete. Saved %s" % BLEND_OUT)
    return sc

def setup_render(sc, preset):
    scn = sc
    scn.render.engine = 'CYCLES'
    scn.cycles.device = 'CPU'
    scn.render.resolution_x = preset["w"]
    scn.render.resolution_y = preset["h"]
    scn.render.resolution_percentage = 100
    scn.cycles.samples = preset["samples"]
    scn.cycles.max_bounces = preset["max_bounces"]
    scn.cycles.diffuse_bounces = 2
    scn.cycles.glossy_bounces = 1
    scn.cycles.transmission_bounces = 1
    scn.cycles.transparent_max_bounces = 2
    scn.cycles.use_denoising = False
    scn.cycles.seed = preset["seed"]
    scn.cycles.use_adaptive_sampling = False
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGB'
    scn.render.image_settings.compression = 80
    scn.view_settings.exposure = -0.9
    return scn

# ============================================================================
# BENCHMARK / RENDER (resume-safe)
# ============================================================================
BENCH_FRAMES = [48, 200, 285, 425, 590, 760, 905, 1125]

def bench_one(sc, n, outdir):
    sc.frame_set(n)
    sc.render.filepath = str(frame_path(outdir, n))
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    dt = time.time() - t0
    log_line("BENCHMARK frame %d: %.2fs" % (n, dt))
    return dt

def run_benchmark(sc):
    BENCH_DIR.mkdir(exist_ok=True)
    preset = PRESETS[FORCE_PRESET or "A"]
    setup_render(sc, preset)
    times = []
    for n in BENCH_FRAMES:
        times.append(bench_one(sc, n, BENCH_DIR))
    avg = sum(times) / len(times)
    log_line("Average per frame: %.3fs over %d samples" % (avg, len(times)))
    est_min = avg * FRAME_END / 60.0
    log_line("Estimated total render: %.1f minutes (%d frames @ 8 fps grid)" % (est_min, FRAME_END))
    if FORCE_PRESET:
        log_line("FORCED PRESET %s (--preset)" % FORCE_PRESET)
        return FORCE_PRESET, est_min
    for name in ("A", "B", "C"):
        est = avg * FRAME_END / 60.0 * (PRESETS[name]["samples"] / PRESETS["A"]["samples"])
        if est <= BUDGET_MIN:
            log_line("SELECTED PRESET %s — estimated %.1f min <= %d min budget." % (name, est, BUDGET_MIN))
            return name, est
    log_line("FATAL: even PRESET C exceeds the %d minute budget — refusing to render." % BUDGET_MIN)
    sys.exit(3)

def render_loop(sc, preset_name):
    preset = PRESETS[preset_name]
    setup_render(sc, preset)
    FRAMES_DIR.mkdir(exist_ok=True)
    total = 0; t_start = time.time()
    for n in range(1, FRAME_END + 1):
        p = frame_path(FRAMES_DIR, n)
        if frame_is_valid(p):
            continue
        sc.frame_set(n)
        sc.render.filepath = str(p)
        t0 = time.time()
        try:
            bpy.ops.render.render(write_still=True)
        except Exception as e:
            log_line("CRASH at frame %d: %s — saving blend and aborting (resume with --render)" % (n, e))
            try:
                bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT), check_existing=False)
            except Exception:
                pass
            sys.exit(4)
        dt = time.time() - t0
        total += 1
        log_line("RENDER frame %d/%d %.2fs" % (n, FRAME_END, dt))
        if n % 200 == 0:
            try:
                bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT), check_existing=False)
                log_line("Incremental save -> %s" % BLEND_OUT)
            except Exception:
                pass
    mins = (time.time() - t_start) / 60.0
    log_line("RENDER COMPLETE: %d frames in %.1f minutes" % (total, mins))
    try:
        bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUT), check_existing=False)
    except Exception:
        pass
    return True

# ============================================================================
# POST: print-grade (dither / split-tone / halftone / CA / grain / vignette)
# ============================================================================
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32) / 16.0

def bayer_tile(h, w):
    return np.tile(BAYER4, ((h + 3) // 4, (w + 3) // 4))[:h, :w]

def bilinear_sample(img, yy, xx):
    h, w = img.shape[:2]
    x0 = np.floor(xx).astype(np.int32); y0 = np.floor(yy).astype(np.int32)
    x1 = x0 + 1; y1 = y0 + 1
    dx = xx - x0; dy = yy - y0
    def cl(i, m): return np.clip(i, 0, m - 1)
    Ia = img[cl(y0, h), cl(x0, w)] * (1 - dx)[..., None] * (1 - dy)[..., None]
    Ib = img[cl(y0, h), cl(x1, w)] * dx[..., None] * (1 - dy)[..., None]
    Ic = img[cl(y1, h), cl(x0, w)] * (1 - dx)[..., None] * dy[..., None]
    Id = img[cl(y1, h), cl(x1, w)] * dx[..., None] * dy[..., None]
    return Ia + Ib + Ic + Id

def grade_frame(arr, n):
    """arr: float32 HxWx3 in [0,1], top-down. Deterministic per frame index."""
    h, w = arr.shape[:2]
    lum = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # --- split-tone: shadows -> teal, highlights -> warm amber ---
    shadow = np.clip(1.0 - lum / 0.45, 0.0, 1.0) ** 1.6
    hi = np.clip((lum - 0.55) / 0.45, 0.0, 1.0) ** 1.5
    arr[:, :, 0] -= shadow * 0.045; arr[:, :, 1] += shadow * 0.018; arr[:, :, 2] += shadow * 0.055
    arr[:, :, 0] += hi * 0.060;     arr[:, :, 1] += hi * 0.016;   arr[:, :, 2] -= hi * 0.065
    # --- gentle S-curve contrast ---
    arr = np.clip((arr - 0.5) * 1.09 + 0.5, 0.0, 1.0)
    lum = arr @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # --- Bayer 4x4 ordered dither, quantize to 36 levels ---
    B = bayer_tile(h, w)[..., None]
    L = 36.0
    arr = (np.floor(arr * L + B - 0.5) + 0.5) / L
    # --- coarse halftone in illuminated regions (dot grid) ---
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    period = 5.0
    strength = np.clip((lum - 0.60) / 0.40, 0.0, 1.0) ** 1.4
    grid = ((xx % period) - period / 2) ** 2 + ((yy % period) - period / 2) ** 2
    radius = 1.1 + 4.6 * strength
    dotmask = (grid < radius ** 2).astype(np.float32) * strength
    arr *= (1.0 - 0.22 * dotmask[..., None])
    # --- radial chromatic aberration (<=1.4 px at corners) ---
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    dxn = (xx - cx) / cx; dyn = (yy - cy) / cy
    r = np.sqrt(dxn * dxn + dyn * dyn)
    ca = 1.35 * (r ** 2.4)
    ux = np.where(r > 1e-4, dxn / np.maximum(r, 1e-4), 0.0); uy = np.where(r > 1e-4, dyn / np.maximum(r, 1e-4), 0.0)
    Rr = bilinear_sample(arr[:, :, 0:1], yy - uy * ca, xx - ux * ca)[:, :, 0]
    Bb = bilinear_sample(arr[:, :, 2:3], yy + uy * ca, xx + ux * ca)[:, :, 0]
    arr[:, :, 0] = Rr; arr[:, :, 2] = Bb
    # --- animated film grain (luminance-aware) ---
    rng = np.random.default_rng(91000 + n * 37)
    grain = (rng.standard_normal((h, w, 1)) * 0.035) * (0.35 + 0.65 * (1.0 - np.abs(lum[..., None] - 0.5) * 2.0))
    arr = arr + grain
    # --- vignette ---
    vig = 1.0 - 0.34 * (r ** 2.05)
    arr = arr * vig[..., None]
    return np.clip(arr, 0.0, 1.0).astype(np.float32)

def load_png_rgb(path):
    img = bpy.data.images.load(str(path), check_existing=False)
    img.colorspace_settings.name = 'Non-Color'
    w, h = img.size
    arr = np.asarray(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[..., :3][::-1]
    bpy.data.images.remove(img)
    return arr

def save_png_rgb(path, arr):
    h, w = arr.shape[:2]
    rgba = np.concatenate([arr[::-1], np.ones((h, w, 1), dtype=np.float32)], axis=2).ravel()
    img = bpy.data.images.new("WRITE_%s" % path.name, w, h, alpha=True)
    img.colorspace_settings.name = 'Non-Color'
    img.pixels = rgba.tolist()
    img.file_format = 'PNG'
    img.filepath_raw = str(path)
    img.save()
    bpy.data.images.remove(img)

def title_alpha(t):
    up = min(1.0, max(0.0, (t - 152.0) / 2.0))
    down = min(1.0, max(0.0, (164.2 - t) / 3.2))
    return min(up, down)

def cmd_post():
    import glob as _glob
    log_section("POST: PRINT GRADE (Bayer dither / split-tone / halftone / CA / grain / vignette)")
    POST_DIR.mkdir(exist_ok=True)
    title_path = POST_DIR / "_title_card.png"
    if not title_path.exists():
        ff = next(iter(_glob.glob(str(_ENV / "pylibs" / "imageio_ffmpeg" / "binaries" / "ffmpeg-linux64-*"))), None)
        if ff:
            import subprocess
            font = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
            subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi",
                            "-i", "color=c=0x00000000:s=640x270,format=rgba",
                            "-vf", "drawtext=fontfile=%s:text='THE LAST MEMORY OF RAIN':fontcolor=0xEDE3C8:fontsize=15:x=(w-text_w)/2:y=h-52" % font,
                            "-frames:v", "1", str(title_path)], check=False)
    title_arr = None
    if title_path.exists():
        timg = bpy.data.images.load(str(title_path), check_existing=False)
        timg.colorspace_settings.name = 'Non-Color'
        tw, th = timg.size
        tarr = np.asarray(timg.pixels[:], dtype=np.float32).reshape(th, tw, 4)[::-1]
        bpy.data.images.remove(timg)
        title_arr = tarr
        log_line("End-card title rendered (%dx%d)" % (tw, th))
    for n in range(1, FRAME_END + 1):
        src = frame_path(FRAMES_DIR, n)
        if not frame_is_valid(src):
            log_line("POST skip (missing source): %s" % src); continue
        dst = frame_path(POST_DIR, n)
        arr = load_png_rgb(src)
        arr = grade_frame(arr, n)
        if title_arr is not None:
            t = (n - 1) / FPS_RENDER
            a = title_alpha(t)
            if a > 0.004:
                ah, aw = title_arr.shape[:2]
                h, w = arr.shape[:2]
                crop = title_arr[:min(ah, h), :min(aw, w)]
                alpha = crop[..., 3:4] * a
                arr = arr * (1.0 - alpha) + crop[..., :3] * alpha
        save_png_rgb(dst, arr)
        if n % 200 == 0:
            log_line("POST frame %d/%d" % (n, FRAME_END))
    log_line("POST COMPLETE -> %s" % POST_DIR)

# ============================================================================
# ASSEMBLY: 8fps grid -> 24fps triplication -> 1920x810 nearest -> H.264/AAC
# ============================================================================
def ffmpeg_bin():
    import glob as _glob
    hits = _glob.glob(str(_ENV / "pylibs" / "imageio_ffmpeg" / "binaries" / "ffmpeg-linux64-*"))
    if not hits:
        raise RuntimeError("ffmpeg binary not found in _env — run restore_env.sh")
    return hits[0]

def cmd_assemble():
    import subprocess
    ff = ffmpeg_bin()
    log_section("ASSEMBLY: 1920x810 / 24 fps / H.264 + AAC")
    frames = sorted(POST_DIR.glob("frame_*.png"))
    if len(frames) < FRAME_END:
        log_line("FATAL: only %d/%d post frames present — refusing to assemble a short film" % (len(frames), FRAME_END))
        sys.exit(5)
    common_audio = ["-i", str(AUDIO_WAV), "-map", "0:v:0", "-map", "1:a:0"]
    base = [ff, "-y", "-loglevel", "error", "-framerate", str(FPS_RENDER),
            "-i", str(POST_DIR / "frame_%04d.png")]
    cmd = base + common_audio + [
        "-vf", "fps=24,scale=1920:810:flags=neighbor",
        "-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
        "-crf", "17", "-preset", "slow", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", "-shortest", str(FINAL_MP4)]
    log_line("FFMPEG final: %s" % " ".join(cmd))
    subprocess.run(cmd, check=True)
    cmd2 = base + common_audio + [
        "-vf", "fps=24,scale=1280:540:flags=neighbor",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(PREVIEW_MP4)]
    subprocess.run(cmd2, check=True)
    log_line("ASSEMBLY COMPLETE -> %s (%.2f MB)  preview -> %s" % (
        FINAL_MP4.name, FINAL_MP4.stat().st_size / 1e6, PREVIEW_MP4.name))

# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--preset", choices=["A", "B", "C"], default=None)
    args = ap.parse_args()
    global FORCE_PRESET
    FORCE_PRESET = args.preset
    do_all = args.all
    sc = build_scene()
    if args.build_only:
        log_line("--build-only: scene built and saved to %s" % BLEND_OUT); return
    if args.benchmark or do_all:
        chosen, est = run_benchmark(sc)
        preset_name = FORCE_PRESET or chosen
    else:
        preset_name = FORCE_PRESET or "A"
        log_line("No benchmark requested; using preset %s" % preset_name)
    if args.render or do_all:
        if not render_loop(sc, preset_name):
            return
    if args.post or do_all:
        cmd_post()
    if args.assemble or do_all:
        cmd_assemble()
    log_section("DONE")

if __name__ == "__main__":
    main()
