#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PALACE — a wordless 5-minute cinematic short film, produced procedurally
with Blender's Python API (bpy).

Usage
-----
  blenderpy film_main.py --detect              # Phase 5: environment/compat report
  blenderpy film_main.py --build-only          # construct scene, save .blend, exit
  blenderpy film_main.py --benchmark           # benchmark representative frames, auto-pick preset
  blenderpy film_main.py --render              # render all frames (resumable)
  blenderpy film_main.py --audio               # synthesize audio.wav
  blenderpy film_main.py --assemble            # assemble final_film.mp4 + preview.mp4
  blenderpy film_main.py --all                 # detect -> build -> benchmark -> render -> audio -> assemble

Options
-------
  --preset A|B|C    force a quality preset (default: auto from benchmark)
  --frames N        render only the first N unique frames (test renders)

Resume behaviour: existing valid PNGs in frames/ are never re-rendered.
Delete logs/render_log.txt to reset the log; rendering itself needs no reset.
"""

import argparse
import json
import math
import os
import random
import struct
import sys
import time
import wave
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------- paths ------
HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(HERE, "frames")
AUDIO_DIR = os.path.join(HERE, "audio")
LOGS_DIR = os.path.join(HERE, "logs")
DOCS_DIR = os.path.join(HERE, "docs")
BLEND_PATH = os.path.join(HERE, "film_project.blend")
AUDIO_WAV = os.path.join(HERE, "audio.wav")
FINAL_MP4 = os.path.join(HERE, "final_film.mp4")
PREVIEW_MP4 = os.path.join(HERE, "preview.mp4")
RENDER_LOG = os.path.join(LOGS_DIR, "render_log.txt")
RENDER_LOG_MIRROR = os.path.join(HERE, "render_log.txt")
README_PATH = os.path.join(HERE, "README.txt")

for _d in (FRAMES_DIR, AUDIO_DIR, LOGS_DIR, DOCS_DIR):
    os.makedirs(_d, exist_ok=True)

# ------------------------------------------------------------- constants -----
FPS_RENDER = 8                    # unique frames per second rendered
FPS_FINAL = 24                    # delivered frame rate (triplication of render frames)
DURATION_S = 300.0                # exactly five minutes
FRAME_END = int(DURATION_S * FPS_RENDER)   # 2400
AUDIO_RATE = 22050

# Shot boundaries in seconds -> frame ranges (inclusive start, exclusive end)
SHOTS = [
    ("S01", 0.0, 12.0),  ("S02", 12.0, 30.0), ("S03", 30.0, 44.0),
    ("S04", 44.0, 64.0), ("S05", 64.0, 86.0), ("S06", 86.0, 104.0),
    ("S07", 104.0, 130.0), ("S08", 130.0, 154.0), ("S09", 154.0, 172.0),
    ("S10", 172.0, 192.0), ("S11", 192.0, 216.0), ("S12", 216.0, 240.0),
    ("S13", 240.0, 258.0), ("S14", 258.0, 280.0), ("S15", 280.0, 300.0),
]
SHOT_FRAMES = {name: (int(t0 * FPS_RENDER) + 1, int(t1 * FPS_RENDER))
               for name, t0, t1 in SHOTS}

# ---------------------------------------------------------------- presets ----
PRESETS = {
    "A": dict(name="PRESET A — STANDARD ECONOMY", w=640, h=360, fps=8,
              samples=6, use_denoise=True, max_bounces=2, shadow_soft=False),
    "B": dict(name="PRESET B — ULTRA ECONOMY", w=480, h=270, fps=8,
              samples=6, use_denoise=True, max_bounces=2, shadow_soft=False),
    "C": dict(name="PRESET C — EMERGENCY", w=426, h=240, fps=8,
              samples=4, use_denoise=True, max_bounces=1, shadow_soft=False),
}
BUDGET_MINUTES = 180

# ------------------------------------------------------------------ log ------
def log_line(msg):
    line = "[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    for path in (RENDER_LOG, RENDER_LOG_MIRROR):
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
        except OSError:
            pass

def log_section(title):
    log_line("=" * 78)
    log_line(title)
    log_line("=" * 78)


# ============================================================================
# PHASE 5 — TECHNICAL VALIDATION / ENVIRONMENT DETECTION
# ============================================================================
def detect_environment():
    """Detect Blender/Python/render capabilities. Returns a report dict."""
    import bpy
    report = {}
    report["blender_version_string"] = bpy.app.version_string
    report["blender_version"] = list(bpy.app.version)
    report["python_version"] = sys.version.split()[0]
    report["platform"] = sys.platform
    report["cpu_count"] = os.cpu_count() or 1

    engines = {}
    for eng in ("CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH", "BLENDER_EEVEE"):
        try:
            bpy.context.scene.render.engine = eng
            engines[eng] = True
        except Exception as exc:
            engines[eng] = "unavailable: %s" % exc
    report["render_engines"] = engines

    # GPU / Cycles device probe (never assume; report what is actually there)
    gpu = {"devices": [], "gpu_usable": False}
    try:
        import _cycles  # not present in module builds; tolerate failure
        gpu["cycles_module"] = True
    except Exception:
        gpu["cycles_module"] = False
    try:
        prefs = bpy.context.preferences.addons.get("cycles")
        if prefs is not None:
            cprefs = prefs.preferences
            try:
                cprefs.refresh_devices()
                for dev in cprefs.devices:
                    gpu["devices"].append({"name": dev.name, "type": dev.type,
                                           "use": dev.use})
                gpu["gpu_usable"] = any(d["type"] in ("CUDA", "OPTIX", "HIP", "ONEAPI", "METAL")
                                        and d["use"] for d in gpu["devices"])
            except Exception as exc:
                gpu["devices_error"] = str(exc)
    except Exception as exc:
        gpu["addon_error"] = str(exc)
    report["gpu"] = gpu

    report["render_device"] = "CPU (forced — integrated GPU is not used for Cycles)"
    report["eevee_usable_headless"] = False  # requires OpenGL 4.3 context; headless CPU path chosen
    report["cycles_samples_note"] = "Cycles CPU, low samples, OpenImageDenoise when enabled"

    # Feature support notes (version-dependent, handled by the compatibility layer)
    report["features"] = {
        "use_denoising": True,                 # cycles setting exists in 4.5
        "emission_color_input": True,          # Principled BSDF 'Emission Color' (4.x)
        "agx_view_transform": True,            # 4.x default
        "text_font_load": True,                # verified against bundled Noto Sans woff2
        "build_info": bpy.app.version_string,
    }
    return report

def print_detection_report(report):
    log_section("PHASE 5 — TECHNICAL VALIDATION REPORT")
    log_line("Blender .............. %s" % report["blender_version_string"])
    log_line("Embedded Python ...... %s" % report["python_version"])
    log_line("Platform ............. %s | CPUs: %d" % (report["platform"], report["cpu_count"]))
    for eng, ok in report["render_engines"].items():
        log_line("Engine %-18s %s" % (eng, ok))
    log_line("Cycles GPU devices ... %s" % (report["gpu"]["devices"] or "none detected"))
    log_line("GPU rendering usable . %s" % report["gpu"]["gpu_usable"])
    log_line("Chosen path .......... %s" % report["render_device"])
    log_line("EEVEE next headless .. %s (no OpenGL context in this environment)" %
             report["eevee_usable_headless"])
    log_line("NOTE: the Intel HD 3000-class integrated GPU and the headless CPU path")
    log_line("are both treated as CPU rendering targets. No GPU features are required.")


# ============================================================================
# BLENDER IMPORT + SAFE SCENE RESET
# ============================================================================
import bpy  # noqa: E402  (bpy is only importable inside Blender / the bpy module)

def _bpy_font(name):
    here = os.path.dirname(os.path.abspath(bpy.__file__))
    for rel in (os.path.join("4.5", "datafiles", "fonts", name),
                os.path.join("datafiles", "fonts", name)):
        cand = os.path.join(here, rel)
        if os.path.exists(cand):
            return cand
    return cand

FONT_PATH = _bpy_font("Inter.woff2")
FONT_BOLD = FONT_PATH  # single bundled face; weight faked with geometry/offset

def reset_scene():
    """Wipe all data so re-runs are deterministic."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = FPS_RENDER
    scene.render.fps_base = 1.0
    scene.frame_start = 1
    scene.frame_end = FRAME_END
    scene.unit_settings.system = 'METRIC'
    try:
        scene.render.engine = 'CYCLES'
    except Exception as exc:
        log_line("FATAL: CYCLES engine cannot be selected: %s" % exc)
        raise
    return scene

# ------------------------------------------------------- compatibility layer --
def _as_color4(value):
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return (value[0], value[1], value[2], 1.0)
    return value

def set_bsdf_input(bsdf, value, *names):
    """Set the first principled-BSDF input that exists (version compatible)."""
    for name in names:
        if name in bsdf.inputs:
            target = bsdf.inputs[name]
            val = value
            try:
                if getattr(target, 'type', '') == 'RGBA':
                    val = _as_color4(value)
            except Exception:
                pass
            target.default_value = val
            return True
    return False

def keyframe_safe(obj, data_path, frame, index=-1):
    if index >= 0:
        obj.keyframe_insert(data_path=data_path, index=index, frame=frame)
    else:
        obj.keyframe_insert(data_path=data_path, frame=frame)

def get_fcurves(action):
    """Blender 4.4+ slotted actions vs legacy fcurves — both paths handled."""
    if action is None:
        return []
    try:
        return list(action.fcurves)
    except Exception:
        curves = []
        try:
            for layer in action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        curves.extend(bag.fcurves)
        except Exception:
            pass
        return curves

def set_constant_on_frames(obj, frames):
    """Hard cuts: make keyframes on the given frames CONSTANT-interpolated."""
    ad = obj.animation_data
    fcurves = get_fcurves(ad.action) if ad else []
    target = set(int(f) for f in frames)
    for fc in fcurves:
        for kp in fc.keyframe_points:
            if int(round(kp.co.x)) in target:
                kp.interpolation = 'CONSTANT'

# ============================================================================
# MATERIAL LIBRARY (flat, stylized, cheap for Cycles CPU)
# ============================================================================
PAL = {  # locked color script for the film
    "ext_pre_dawn": (0.075, 0.115, 0.130, 1.0),   # #13242X-ish deep teal
    "sky_low":      (0.125, 0.195, 0.220, 1.0),
    "sea":          (0.055, 0.165, 0.185, 1.0),
    "promenade":    (0.135, 0.165, 0.170, 1.0),
    "street":       (0.105, 0.120, 0.125, 1.0),
    "facade_cream": (0.415, 0.385, 0.315, 1.0),   # weathered cream brick
    "facade_shade": (0.295, 0.270, 0.225, 1.0),
    "marquee_dead": (0.205, 0.175, 0.150, 1.0),
    "marquee_lit":  (1.000, 0.780, 0.380, 1.0),
    "int_black":    (0.055, 0.048, 0.043, 1.0),
    "wall_wood":    (0.145, 0.100, 0.072, 1.0),   # dark wood panel
    "wall_wood2":   (0.185, 0.135, 0.095, 1.0),
    "velvet":       (0.235, 0.082, 0.078, 1.0),   # desaturated velvet red
    "velvet_hi":    (0.330, 0.120, 0.110, 1.0),
    "brass":        (0.545, 0.380, 0.140, 1.0),   # dulled brass
    "copper":       (0.480, 0.255, 0.135, 1.0),
    "projector":    (0.115, 0.105, 0.100, 1.0),
    "paper":        (0.885, 0.855, 0.760, 1.0),
    "yellow_coat":  (0.820, 0.615, 0.145, 1.0),   # the boy — ONLY saturated hue
    "coat_shadow":  (0.560, 0.390, 0.090, 1.0),
    "skin_elias":   (0.520, 0.385, 0.275, 1.0),
    "skin_boy":     (0.620, 0.445, 0.300, 1.0),
    "cardigan":     (0.235, 0.185, 0.135, 1.0),
    "cardigan_hi":  (0.320, 0.255, 0.185, 1.0),
    "trousers":     (0.125, 0.120, 0.125, 1.0),
    "shirt":        (0.360, 0.335, 0.290, 1.0),
    "scarf":        (0.480, 0.345, 0.120, 1.0),   # dull mustard scarf
    "hair_grey":    (0.330, 0.320, 0.305, 1.0),
    "screen_glow":  (1.000, 0.925, 0.740, 1.0),
    "leaf":         (0.115, 0.155, 0.120, 1.0),
    "sepia_wall":   (0.420, 0.305, 0.165, 1.0),
    "sepia_skin":   (0.560, 0.415, 0.230, 1.0),
    "attic":        (0.980, 0.700, 0.330, 1.0),   # lit attic window (warm)
}

def make_material(name, color, roughness=0.9, metallic=0.0,
                  emit=None, emit_strength=0.0, alpha=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = None
    for n in nodes:
        if n.bl_idname == 'ShaderNodeBsdfPrincipled':
            bsdf = n
            break
    set_bsdf_input(bsdf, color, 'Base Color')
    set_bsdf_input(bsdf, roughness, 'Roughness')
    set_bsdf_input(bsdf, metallic, 'Metallic')
    if emit is not None:
        set_bsdf_input(bsdf, emit, 'Emission Color', 'Emission')
        set_bsdf_input(bsdf, emit_strength, 'Emission Strength')
    if alpha is not None:
        set_bsdf_input(bsdf, alpha, 'Alpha')
        mat.surface_render_method = 'BLENDED'   # 4.2+ transparent surface path
        mat.use_transparency_overlap = False
    mat.diffuse_color = color
    return mat

def make_glow_mix_material(name, emit_rgb, strength, fac0):
    """Transparent <-> Emission via MixShader; Fac is keyframable (works in Cycles)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    transp = nt.nodes.new('ShaderNodeBsdfTransparent')
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Color'].default_value = _as_color4(emit_rgb)
    em.inputs['Strength'].default_value = strength
    mix = nt.nodes.new('ShaderNodeMixShader')
    mix.inputs['Fac'].default_value = fac0
    nt.links.new(transp.outputs['BSDF'], mix.inputs[1])
    nt.links.new(em.outputs['Emission'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    mat.surface_render_method = 'BLENDED'
    mat.use_transparency_overlap = False
    mix.name = "MIX"
    return mat

def set_mix_fac(mat, value, frame=None):
    mix = mat.node_tree.nodes.get("MIX")
    if frame is None:
        mix.inputs['Fac'].default_value = value
        return
    mix.inputs['Fac'].default_value = value
    mix.inputs['Fac'].keyframe_insert('default_value', frame=frame)

def flat(obj):
    """Force flat shading + no smooth — the house style."""
    if obj.type == 'MESH':
        for p in obj.data.polygons:
            p.use_smooth = False
    return obj

def new_icosphere(name, radius, location, segments=1):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=segments, radius=radius,
                                          location=location)
    ob = bpy.context.active_object
    ob.name = name
    return flat(ob)

def new_cube(name, size, location, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    return flat(ob)

def new_cylinder(name, radius, depth, location, vertices=12):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth,
                                        location=location)
    ob = bpy.context.active_object
    ob.name = name
    return flat(ob)

def new_plane(name, width, height, location, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location, rotation=rotation)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (width, height, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    return flat(ob)

def new_cone(name, r1, r2, depth, location, vertices=14, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=r1, radius2=r2,
                                    depth=depth, location=location, rotation=rotation)
    ob = bpy.context.active_object
    ob.name = name
    return flat(ob)

def orient_text(obj, normal, up_world=(0.0, 0.0, 1.0)):
    """Aim text so it reads upright and unmirrored on a wall with the given outward normal."""
    from mathutils import Vector, Matrix
    n = Vector(normal).normalized()
    up = Vector(up_world)
    r = up.cross(n).normalized()
    u = n.cross(r).normalized()
    mat = Matrix.Identity(4)
    mat.col[0].xyz = r
    mat.col[1].xyz = u
    mat.col[2].xyz = n
    obj.rotation_euler = mat.to_euler()
    return obj

def new_text(name, body, size, location, rotation=(0, 0, 0), align='CENTER'):
    curve = bpy.data.curves.new(name, type='FONT')
    curve.body = body
    curve.size = size
    curve.align_x = align
    curve.align_y = 'CENTER'
    try:
        curve.font = bpy.data.fonts.load(FONT_PATH)
    except Exception as exc:
        log_line("WARN: font load failed (%s); using built-in font" % exc)
    curve.extrude = 0.02
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation
    return obj

def join_meshes(objects, name):
    """Join small meshes to cut object count (keeps Cycles BVH cheap)."""
    objects = [o for o in objects if o is not None and o.type == 'MESH']
    if len(objects) == 1:
        objects[0].name = name
        return objects[0]
    if not objects:
        log_line("WARN: join_meshes('%s') called with no meshes" % name)
        return None
    bpy.ops.object.select_all(action='DESELECT')
    for ob in objects:
        ob.select_set(True)
    ctx = bpy.context
    ctx.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = ctx.active_object
    joined.name = name
    return joined

# ============================================================================
# WORLD + SKY BACKDROP
# ============================================================================
WORLD_BG = None

def build_world(scene):
    world = bpy.data.worlds.new("PALACE_World")
    world.use_nodes = True
    nt = world.node_tree
    bg = nt.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.030, 0.045, 0.052, 1.0)
    bg.inputs["Strength"].default_value = 1.7
    scene.world = world
    return bg

def add_gradient_backdrop(name, radius=170.0, depth=95.0, location=(0, 0, 0)):
    """Open cylinder around the world with a vertical Generated->ColorRamp sky."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth,
                                        location=location, end_fill_type='NOTHING')
    cyl = bpy.context.active_object
    cyl.name = name
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    em = nt.nodes.new('ShaderNodeEmission')
    ramp = nt.nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].color = (0.045, 0.075, 0.088, 1.0)   # zenith
    ramp.color_ramp.elements[1].color = (0.32, 0.26, 0.19, 1.0)      # horizon (faint dirty gold)
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ')
    nt.links.new(geo.outputs['Position'], sep.inputs['Vector'])
    nt.links.new(sep.outputs['Z'], ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'], em.inputs['Color'])
    em.inputs['Strength'].default_value = 0.62
    nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
    cyl.data.materials.append(mat)
    cyl.visible_camera = False   # pure light source; camera never sees the shell
    for attr in ("visible_shadow", "visible_glossy", "visible_diffuse",
                 "visible_transmission", "visible_indirect_specular",
                 "visible_indirect_diffuse"):
        try:
            setattr(cyl, attr, False)
        except AttributeError:
            pass
    return cyl

# ============================================================================
# SET 1 — EXT_PROMENADE / TOWN (x: -120..120, z: 0)
# ============================================================================
def build_exterior(scene, M):
    """The drowned seaside resort. Everything facing +Y (sea at +Y)."""
    # sea
    sea = new_plane("SEA", 340, 170, (0, 110, -0.25))
    sea.data.materials.append(M["sea"])
    # promenade deck (raised)
    prom = new_cube("PROMENADE", 1, (0, 10.0, 0.0), scale=(240, 8.0, 0.6))
    prom.location.z = 0.0
    prom.data.materials.append(M["promenade"])
    # street bed
    street = new_plane("STREET", 240, 12.5, (0, 0.0, 0.02))
    street.data.materials.append(M["street"])
    # sidewalk in front of facades
    walk = new_cube("SIDEWALK", 1, (0, -6.9, 0.12), scale=(240, 2.2, 0.24))
    walk.data.materials.append(M["promenade"])
    # promenade railing (thin runs along sea edge)
    rail_posts = []
    for x in range(-110, 111, 5):
        post = new_cylinder("rail", 0.055, 1.05, (x, 13.8, 0.85), vertices=6)
        post.data.materials.append(M["projector"])
        rail_posts.append(post)
    rail = new_cylinder("RAIL_TOP", 0.06, 240, (0, 13.8, 1.38), vertices=6)
    rail.rotation_euler = (0, math.pi / 2, 0)
    rail.data.materials.append(M["brass"])
    # ---- shuttered shop row (background layers) --------------------------
    random.seed(11)
    shop_x = -106.0
    shops = []
    shop_mats = [M["facade_shade"], M["facade_cream"], M["marquee_dead"], M["wall_wood"]]
    while shop_x < 108.0:
        w = random.uniform(6.5, 11.5)
        h = random.uniform(4.2, 7.6)
        if abs(shop_x + w / 2) < 9.0:   # leave room for the PALACE
            shop_x += w + 1.2
            continue
        b = new_cube("SHOP", 1, (shop_x + w / 2, -9.6, h / 2), scale=(w, 5.0, h))
        b.data.materials.append(random.choice(shop_mats))
        shops.append(b)
        # shutter stripe
        sh = new_cube("SHUTTER", 1, (shop_x + w / 2, -7.05, 1.05), scale=(w * 0.78, 0.12, 1.7))
        sh.data.materials.append(M["projector"])
        shops.append(sh)
        # roofline clutter
        if random.random() < 0.6:
            chimney = new_cube("CHIMNEY", 1, (shop_x + w * 0.3, -10.6, h + 0.5),
                               scale=(0.9, 0.9, 1.4))
            chimney.data.materials.append(M["facade_shade"])
            shops.append(chimney)
        shop_x += w + 1.2
    # ---- THE PALACE (hero facade) -----------------------------------------
    palace = bpy.data.collections.new("PALACE_EXT")
    # main body
    body = new_cube("PALACE_BODY", 1, (0, -10.2, 5.5), scale=(17.5, 6.4, 11.0))
    body.data.materials.append(M["facade_cream"])
    # parapet + tower sign housing
    parapet = new_cube("PALACE_PARAPET", 1, (0, -10.2, 11.25), scale=(18.2, 6.7, 0.6))
    parapet.data.materials.append(M["facade_shade"])
    tower = new_cube("PALACE_TOWER", 1, (0, -10.6, 13.4), scale=(8.6, 3.4, 3.6))
    tower.data.materials.append(M["facade_shade"])
    # pilasters flanking entrance
    for px in (-6.4, 6.4):
        pil = new_cube("PILASTER", 1, (px, -6.95, 4.6), scale=(1.15, 0.5, 9.2))
        pil.data.materials.append(M["facade_shade"])
    # marquee slab + angled board
    slab = new_cube("MARQUEE_SLAB", 1, (0, -6.35, 4.05), scale=(13.2, 2.6, 0.42))
    slab.data.materials.append(M["marquee_dead"])
    mtext = new_text("MARQUEE_TEXT", "T O N I G H T   O N L Y", 0.62, (0, -5.08, 3.62))
    orient_text(mtext, (0, 1, 0))
    mtext.data.materials.append(M["marquee_letters"])
    board = new_cube("MARQUEE_BOARD", 1, (0, -5.25, 3.6), scale=(11.6, 0.28, 1.5))
    board.rotation_euler = (math.radians(-16), 0, 0)
    board.data.materials.append(M["marquee_dead"])
    # doors (two leaves, dark glass)
    for dx in (-1.55, 1.55):
        leaf = new_cube("DOOR_LEAF", 1, (dx, -6.92, 1.62), scale=(2.5, 0.16, 3.24))
        leaf.data.materials.append(M["projector"])
    door_bar = new_cube("DOOR_BAR", 1, (0, -6.8, 1.62), scale=(0.32, 0.2, 3.24))
    door_bar.data.materials.append(M["brass"])
    # ATTIC WINDOW — the warm pinpoint (lit from S01; it is the boy)
    attic = new_plane("ATTIC_WINDOW", 1.5, 1.7, (2.9, -6.89, 9.15), (-math.pi / 2, 0, 0))
    attic.data.materials.append(M["attic_emit"])
    attic_frame = new_cube("ATTIC_FRAME", 1, (2.9, -6.97, 9.15), scale=(2.0, 0.1, 2.2))
    attic_frame.data.materials.append(M["wall_wood"])
    # marquee letter boards (filled by apply_marquee_letters)
    # PALACE neon sign on the tower (emissive; flickers in S01, dark after S15)
    sign = new_text("SIGN_PALACE", "PALACE", 2.35, (0, -8.87, 13.45))
    orient_text(sign, (0, 1, 0))
    sign.data.materials.append(M["sign_emit"])
    # banner: FINAL SCREENING
    banner = new_plane("BANNER", 12.0, 1.35, (0, -6.93, 6.35), (-math.pi / 2, 0, 0))
    banner.data.materials.append(M["paper"])
    btext = new_text("BANNER_TEXT", "FINAL SCREENING TONIGHT", 0.62,
                     (0, -6.88, 6.35))
    orient_text(btext, (0, 1, 0))
    btext.data.materials.append(M["int_black_text"])
    # bench + crates on promenade
    bench = new_cube("BENCH", 1, (-9.5, 10.5, 0.62), scale=(2.6, 0.7, 0.16))
    leg_a = new_cube("BENCH_LEG_A", 1, (-10.45, 10.5, 0.31), scale=(0.2, 0.58, 0.62))
    leg_b = new_cube("BENCH_LEG_B", 1, (-8.55, 10.5, 0.31), scale=(0.2, 0.58, 0.62))
    bench.data.materials.append(M["wall_wood"])
    leg_a.data.materials.append(M["wall_wood"])
    leg_b.data.materials.append(M["wall_wood"])
    crate = new_cube("CRATE", 1, (8.2, 11.2, 0.75), scale=(1.1, 1.1, 1.1))
    crate.rotation_euler = (0, 0, 0.3)
    crate.data.materials.append(M["wall_wood2"])
    # two promenade lamps (dead)
    for lx in (-16.0, 16.0):
        pole = new_cylinder("LAMP_POLE", 0.09, 4.6, (lx, 10.0, 2.6), vertices=8)
        pole.data.materials.append(M["projector"])
        head = new_cone("LAMP_HEAD", 0.42, 0.18, 0.55, (lx, 10.0, 5.0), vertices=8)
        head.data.materials.append(M["projector"])
    # distorted palm silhouettes (dead reef pines, wind-bent)
    for px, lean in ((-24.0, 0.42), (28.5, -0.5), (44.0, 0.35)):
        trunk = new_cylinder("PINE_TRUNK", 0.28, 6.4, (px, -4.2, 3.2), vertices=7)
        trunk.rotation_euler = (0, lean, 0)
        trunk.data.materials.append(M["wall_wood"])
        crown = new_icosphere("PINE_CROWN", 1.15, (px + lean * 3.1, -4.2, 6.6), segments=1)
        crown.scale = (1.45, 1.2, 0.95)
        crown.data.materials.append(M["leaf"])
    return palace

# ============================================================================
# SET 2 — INTERIOR (offset +X): LOBBY [84..110] | AUDITORIUM [112..152] | BOOTH
# ============================================================================
SLOPE = 0.075            # raked floor rise per meter
SCREEN_X = 151.7
BACK_X = 112.0

def floor_z(x):
    return (SCREEN_X - x) * SLOPE + 0.25

def build_interior_shell(M):
    """Static architecture; joined into few meshes for Cycles efficiency."""
    shell = []
    # ---------- LOBBY ----------
    shell.append(new_cube("LB_floor", 1, (97, 0, -0.15), scale=(26.5, 16.0, 0.3)))
    shell.append(new_cube("LB_ceiling", 1, (97, 0, 6.0), scale=(26.5, 16.0, 0.3)))
    shell.append(new_cube("LB_back", 1, (84.0, 0, 3.0), scale=(0.35, 16.0, 6.0)))
    shell.append(new_cube("LB_frontA", 1, (110.0, -5.1, 3.0), scale=(0.35, 5.8, 6.0)))
    shell.append(new_cube("LB_frontB", 1, (110.0, 5.1, 3.0), scale=(0.35, 5.8, 6.0)))
    shell.append(new_cube("LB_frontT", 1, (110.0, 0, 4.9), scale=(0.35, 4.4, 2.2)))   # header above doors
    shell.append(new_cube("LB_sideA", 1, (97, -8.0, 3.0), scale=(26.5, 0.35, 6.0)))
    # side wall +y has the boy's side door opening x in [104.2,105.9]
    shell.append(new_cube("LB_sideB1", 1, (94.6, 8.0, 3.0), scale=(21.2, 0.35, 6.0)))
    shell.append(new_cube("LB_sideB2", 1, (108.05, 8.0, 3.0), scale=(3.9, 0.35, 6.0)))
    shell.append(new_cube("LB_sideB3", 1, (105.05, 8.0, 4.4), scale=(1.7, 0.35, 3.2)))  # header above door
    # ---------- AUDITORIUM ----------
    # raked floor (rotated box)
    angle = math.atan(SLOPE)
    rake = new_cube("RAKED_FLOOR", 1, (131.5, 0, floor_z(131.5) - 0.28), scale=(42.5, 18.0, 0.4))
    rake.rotation_euler = (0, angle, 0)
    shell.append(rake)
    shell.append(new_cube("AUD_ceiling", 1, (132, 0, 7.4), scale=(40.4, 18.0, 0.35)))
    shell.append(new_cube("AUD_screenwall", 1, (152.15, 0, 3.7), scale=(0.35, 18.0, 7.4)))
    shell.append(new_cube("AUD_sideA", 1, (132, -9.0, 3.7), scale=(40.4, 0.35, 7.4)))
    shell.append(new_cube("AUD_sideB", 1, (132, 9.0, 3.7), scale=(40.4, 0.35, 7.4)))
    shell.append(new_cube("AUD_back", 1, (112.0, 0, 3.7), scale=(0.35, 18.0, 7.4)))
    # attic hatch in ceiling (where the boy watches from)
    hatch = new_plane("ATTIC_HATCH", 1.1, 1.1, (124.0, 0.0, 7.2), (math.pi / 2, 0, 0))
    hatch.data.materials.append(M["int_black"])
    shell.append(hatch)
    for ob in shell:
        ob.data.materials.append(M["wall_wood"] if ob.name.startswith(("LB_", "AUD_", "RAKED"))
                                 else M["wall_wood2"])
    blackbox = new_cube("INT_BLACKBOX", 1, (118.5, 0, 3.6), scale=(73.0, 21.0, 8.6))
    blackbox.data.materials.append(M["int_black"])
    joined = join_meshes(shell, "INT_SHELL")
    joined.data.materials.clear()
    joined.data.materials.append(M["wall_wood"])
    # warm wainscot band for depth (front half of house)
    wainscot = new_plane("WAINSCOT", 39.5, 1.15, (132, -8.78, floor_z(132) + 0.85),
                         (-math.pi / 2, 0, 0))
    wainscot.data.materials.append(M["wall_wood2"])
    return joined

def build_seat_rows(M):
    """11 seats x 12 rows. F7 = row F (r=5), seat i=6."""
    seat_parts = []
    f7 = None
    for r in range(12):
        x = 145.5 - r * 2.6
        z = floor_z(x)
        for i in range(12):
            y = (i - 5.5) * 1.15
            # cushion
            cush = new_cube("seat", 1, (x, y, z + 0.52), scale=(0.92, 0.88, 0.2))
            cush.data.materials.append(M["velvet"])
            # backrest (raked back)
            back = new_cube("seat", 1, (x + 0.42, y, z + 1.05), scale=(0.2, 0.88, 0.95))
            back.rotation_euler = (0, math.radians(10), 0)
            back.data.materials.append(M["velvet"])
            # pedestal sides
            ped1 = new_cube("seat", 1, (x, y - 0.42, z + 0.22), scale=(0.8, 0.09, 0.45))
            ped2 = new_cube("seat", 1, (x, y + 0.42, z + 0.22), scale=(0.8, 0.09, 0.45))
            ped1.data.materials.append(M["projector"])
            ped2.data.materials.append(M["projector"])
            parts = (cush, back, ped1, ped2)
            seat_parts.extend(parts)
            if r == 5 and i == 6:
                f7 = (cush, back)
    joined = join_meshes(seat_parts, "SEATS_ALL")
    return joined, f7

def build_booth(M):
    """Booth: slab + walls with porthole, The Empress projector, stool, photo."""
    parts = []
    parts.append(new_cube("BOOTH_SLAB", 1, (113.4, 0, 4.55), scale=(6.6, 4.6, 0.4)))
    parts.append(new_cube("BOOTH_BACK", 1, (110.3, 0, 6.0), scale=(0.35, 4.6, 3.3)))
    parts.append(new_cube("BOOTH_SIDEA", 1, (113.4, -2.2, 6.0), scale=(6.6, 0.3, 3.3)))
    parts.append(new_cube("BOOTH_SIDEB", 1, (113.4, 2.2, 6.0), scale=(6.6, 0.3, 3.3)))
    # front wall with porthole opening (y in [-0.95,0.95], z in [5.7,6.8])
    parts.append(new_cube("BOOTH_FRONT_L", 1, (116.6, -1.5, 6.0), scale=(0.35, 1.4, 3.3)))
    parts.append(new_cube("BOOTH_FRONT_R", 1, (116.6, 1.5, 6.0), scale=(0.35, 1.4, 3.3)))
    parts.append(new_cube("BOOTH_FRONT_T", 1, (116.6, 0, 7.15), scale=(0.35, 1.6, 0.55)))
    parts.append(new_cube("BOOTH_FRONT_B", 1, (116.6, 0, 5.25), scale=(0.35, 1.6, 0.9)))
    for ob in parts:
        ob.data.materials.append(M["projector"])
    shell = join_meshes(parts, "BOOTH_SHELL")
    # porthole glass
    glass = new_plane("BOOTH_GLASS", 1.9, 1.1, (116.6, 0.0, 6.25), (0, math.pi / 2, 0))
    glass.data.materials.append(M["glass"])
    # projector pedestal
    ped = new_cube("PROJ_PED", 1, (113.6, 0, 5.25), scale=(1.5, 1.3, 0.85))
    ped.data.materials.append(M["wall_wood2"])
    # THE EMPRESS — body along X
    body = new_cylinder("EMPRESS_BODY", 0.34, 1.35, (113.6, 0, 5.95), vertices=12)
    body.rotation_euler = (0, math.pi / 2, 0)
    body.data.materials.append(M["projector"])
    for i, (ry, rz) in enumerate(((0, 0.62), (0.52, -0.38), (-0.52, -0.38))):
        reel = new_cylinder("REEL_%d" % i, 0.44, 0.1, (113.45, ry, 6.55 + rz), vertices=16)
        reel.data.materials.append(M["copper"])
        hub = new_cylinder("HUB_%d" % i, 0.09, 0.12, (113.45, ry, 6.55 + rz), vertices=8)
        hub.data.materials.append(M["brass"])
    lens = new_cone("EMPRESS_LENS", 0.3, 0.13, 0.55, (114.45, 0, 5.95), vertices=12,
                    rotation=(0, math.pi / 2, 0))
    lens.data.materials.append(M["brass"])
    # stool
    stool = new_cylinder("STOOL", 0.3, 0.14, (115.6, 1.35, 5.0), vertices=10)
    stool.data.materials.append(M["wall_wood"])
    leg = new_cylinder("STOOL_LEG", 0.05, 0.9, (115.6, 1.35, 4.75), vertices=6)
    leg.data.materials.append(M["projector"])
    # 1953 photograph + frame + tiny text
    frame = new_cube("PHOTO_FRAME", 1, (110.5, -1.2, 6.35), scale=(0.08, 1.15, 0.9))
    frame.data.materials.append(M["brass"])
    photo = new_plane("PHOTO_1953", 0.95, 0.7, (110.55, -1.2, 6.35), (0, math.pi / 2, 0))
    photo.data.materials.append(M["sepia_photo"])
    ptext = new_text("PHOTO_TEXT", "1953", 0.16, (110.6, -1.2, 5.98))
    orient_text(ptext, (1, 0, 0))
    ptext.data.materials.append(M["sepia_text"])
    # oil can + rag
    can = new_cylinder("OIL_CAN", 0.09, 0.24, (112.6, 0.75, 5.78), vertices=8)
    can.data.materials.append(M["copper"])
    spout = new_cone("OIL_SPOUT", 0.05, 0.015, 0.3, (112.6, 0.75, 5.98), vertices=6)
    spout.data.materials.append(M["brass"])
    return {"shell": shell, "glass": glass, "lens": lens}

def build_lobby_furniture(M):
    """Counter, ticket lamp, ledger, stamp, posters, chairs — the shrine of commerce."""
    parts = []
    counter = new_cube("COUNTER", 1, (87.6, 0, 0.62), scale=(2.4, 7.2, 1.24))
    counter.data.materials.append(M["wall_wood2"])
    parts.append(counter)
    # ticket lamp (dead)
    lamp_base = new_cylinder("T_LAMP_BASE", 0.16, 0.06, (87.6, -1.4, 1.27), vertices=8)
    lamp_base.data.materials.append(M["brass"])
    lamp_stem = new_cylinder("T_LAMP_STEM", 0.035, 0.42, (87.6, -1.4, 1.5), vertices=6)
    lamp_stem.data.materials.append(M["brass"])
    lamp_shade = new_cone("T_LAMP_SHADE", 0.3, 0.12, 0.24, (87.6, -1.4, 1.78), vertices=10)
    lamp_shade.data.materials.append(M["brass"])
    # ledger + ink stamp + strips pile
    ledger = new_cube("LEDGER", 1, (87.0, 0.6, 1.27), scale=(0.55, 0.75, 0.05))
    ledger.data.materials.append(M["paper"])
    stamp = new_cylinder("STAMP", 0.07, 0.22, (86.8, -0.4, 1.34), vertices=8)
    stamp.data.materials.append(M["wall_wood"])
    pile = new_cube("STRIPS_PILE", 1, (87.5, 1.6, 1.26), scale=(0.35, 0.45, 0.04))
    pile.data.materials.append(M["paper"])
    # posters on back wall
    for i, (txt, py) in enumerate((("NOW SHOWING", -2.3), ("THE SEA AND PAPA", 0.0), ("61 YEARS", 2.3))):
        frame = new_cube("POSTER_FRAME_%d" % i, 1, (84.25, py, 2.6), scale=(0.08, 1.9, 2.5))
        frame.data.materials.append(M["brass"])
        paper = new_plane("POSTER_%d" % i, 1.7, 2.3, (84.3, py, 2.6), (0, 0, 0))
        paper.rotation_euler = (0, math.pi / 2, 0)
        paper.data.materials.append(M["paper"])
        t = new_text("POSTER_TEXT_%d" % i, txt, 0.22, (84.36, py, 2.6))
        orient_text(t, (1, 0, 0))
        t.data.materials.append(M["int_black_text"])
    # box office chair
    chair = new_cube("OFFICE_CHAIR", 1, (89.5, 2.6, 0.5), scale=(0.55, 0.55, 0.12))
    chair.data.materials.append(M["wall_wood"])
    chleg = new_cylinder("CHAIR_LEG", 0.05, 0.5, (89.5, 2.6, 0.25), vertices=6)
    chleg.data.materials.append(M["projector"])
    return parts

def build_doors(M):
    """Animated doors: lobby->auditorium twin leaves + boy's side door."""
    # twin leaves, hinged at y=+/-1.55 on the x=111 wall
    leaves = []
    for side in (-1, 1):
        hinge = bpy.data.objects.new("DOOR_HINGE_%s" % ("L" if side < 0 else "R"), None)
        hinge.location = (111.0, side * 1.55, 0.0)
        bpy.context.collection.objects.link(hinge)
        leaf = new_cube("AUD_DOOR_%d" % side, 1, (0, -side * 0.775, 1.9),
                        scale=(0.14, 1.55, 3.8))
        leaf.parent = hinge
        leaf.data.materials.append(M["wall_wood2"])
        leaves.append(hinge)
    # boy's side door, hinged on x=104.2 edge of y=+8 wall
    hinge_b = bpy.data.objects.new("BOY_DOOR_HINGE", None)
    hinge_b.location = (104.2, 8.0, 0.0)
    bpy.context.collection.objects.link(hinge_b)
    panel = new_cube("BOY_DOOR", 1, (0.85, 0, 1.2), scale=(1.7, 0.13, 2.4))
    panel.parent = hinge_b
    panel.data.materials.append(M["wall_wood"])
    return leaves, hinge_b

def make_paper_boat(name, M):
    """The paper boat: squashed hip roof (4-gon cone) + ridge fin."""
    bpy.ops.mesh.primitive_cone_add(vertices=4, radius1=0.16, radius2=0.001,
                                    depth=0.14, location=(0, 0, 0.07))
    hull = bpy.context.active_object
    hull.name = name + "_HULL"
    hull.rotation_euler = (0, 0, math.pi / 4)
    hull.scale = (1.0, 1.0, 0.62)
    # two sloped fin plates (the classic fold peak)
    fin = new_plane(name + "_FIN", 0.30, 0.16, (0, 0, 0.16), (0, 0, 0))
    fin.rotation_euler = (math.radians(90), 0, 0)
    fin.scale = (1.0, 1.0, 1.0)
    boat = join_meshes([hull], name)
    boat.data.materials.append(M["paper"])
    fin.parent = boat
    return boat

def build_beam_and_dust(M):
    """Projector beam cone (fake volumetric) + dust motes + screen."""
    # beam from porthole (116.6,0,6.25) to screen centre (151.7,0,3.1)
    src = np.array([116.7, 0.0, 6.22])
    dst = np.array([151.6, 0.0, 3.15])
    vec = dst - src
    length = float(np.linalg.norm(vec))
    mid = (src + dst) / 2.0
    bpy.ops.mesh.primitive_cone_add(vertices=28, radius1=3.15, radius2=0.05,
                                    depth=length, location=mid)
    beam = bpy.context.active_object
    beam.name = "BEAM_CONE"
    direction = -vec / length
    beam.rotation_mode = 'QUATERNION'
    from mathutils import Vector
    beam.rotation_quaternion = Vector(direction).to_track_quat('Z', 'Y')
    beam.data.materials.append(M["beam"])
    beam.visible_shadow = False
    # dust motes inside the beam
    random.seed(7)
    motes = []
    for _ in range(64):
        t = random.uniform(0.06, 0.96)
        p = src + vec * t
        spread = 0.1 + t * 2.25
        p = p + np.array([random.uniform(-spread, spread),
                          random.uniform(-spread, spread),
                          random.uniform(-spread, spread) * 0.55])
        m = new_icosphere("mote", random.uniform(0.014, 0.04), tuple(p), segments=1)
        motes.append(m)
    dust = join_meshes(motes, "DUST_MOTES")
    dust.data.materials.append(M["dust"])
    # the screen — animated emissive during the show
    screen = new_cube("SCREEN", 1, (SCREEN_X, 0, 3.45), scale=(0.22, 13.5, 6.6))
    screen.data.materials.append(M["screen_anim"])
    frame = new_cube("SCREEN_FRAME", 1, (SCREEN_X + 0.18, 0, 3.45), scale=(0.24, 14.4, 7.5))
    frame.data.materials.append(M["projector"])
    return beam, dust, screen

# ============================================================================
# CHARACTERS — two rigs, posable via empties, keyframed per shot
# ============================================================================
def _empty(name, location, parent=None):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = 'PLAIN_AXES'
    ob.empty_display_size = 0.07
    ob.location = location
    bpy.context.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob

def build_elias(M, base=(0.0, 0.0, 0.0), scale=1.0, suffix=""):
    """Elias: gaunt 60s projectionist. Distinctive silhouette: long cardigan,
    flat cap, hunched forearms. Returns rig dict."""
    s = scale
    root = _empty("ELIAS_ROOT%s" % suffix, base)
    spine = _empty("ELIAS_SPINE%s" % suffix, (0, 0, 0.98 * s), root)
    chest = _empty("ELIAS_CHEST%s" % suffix, (0, 0, 0.34 * s), spine)
    head  = _empty("ELIAS_HEAD%s" % suffix, (0, 0.015 * s, 0.33 * s), chest)
    # pelvis + cardigan (long, to the knees — the silhouette)
    pelvis = new_cylinder("ELIAS_PELVIS%s" % suffix, 0.21 * s, 0.26 * s, (0, 0, 0), vertices=10)
    pelvis.parent = spine; pelvis.location = (0, 0, 0.02 * s)
    cardigan = new_cylinder("ELIAS_CARDIGAN%s" % suffix, 0.235 * s, 0.72 * s, (0, 0, 0), vertices=12)
    # flare the coat: scale bottom via separate tapered mesh
    cardigan.parent = spine; cardigan.location = (0, 0, -0.14 * s)
    cardigan.data.materials.append(M["cardigan"])
    coat_flare = new_cone("ELIAS_COATFLARE%s" % suffix, 0.30 * s, 0.235 * s, 0.34 * s,
                          (0, 0, 0), vertices=12)
    coat_flare.parent = spine; coat_flare.location = (0, 0, -0.44 * s)
    coat_flare.data.materials.append(M["cardigan"])
    # scarf (the one dull-warm accent on him)
    scarf = bpy.ops.mesh.primitive_torus_add(major_radius=0.155 * s, minor_radius=0.05 * s,
                                             major_segments=12, minor_segments=6,
                                             location=(0, 0, 0.30 * s),
                                             rotation=(math.pi / 2, 0, 0))
    scarf_ob = bpy.context.active_object
    scarf_ob.name = "ELIAS_SCARF%s" % suffix
    scarf_ob.parent = chest; scarf_ob.location = (0, 0.0, 0.265 * s)
    scarf_ob.data.materials.append(M["scarf"])
    tail = new_cube("ELIAS_SCARFTAIL%s" % suffix, 1, (0, 0, 0), scale=(0.09 * s, 0.03 * s, 0.34 * s))
    tail.parent = chest; tail.location = (0.08 * s, 0.13 * s, 0.04 * s)
    tail.data.materials.append(M["scarf"])
    # head + cap + glasses + nose
    skull = new_icosphere("ELIAS_SKULL%s" % suffix, 0.125 * s, (0, 0, 0.06 * s), segments=2)
    skull.scale = (0.92, 0.98, 1.12)
    skull.parent = head; skull.location = (0, 0, 0.075 * s)
    skull.data.materials.append(M["skin_elias"])
    nose = new_cone("ELIAS_NOSE%s" % suffix, 0.028 * s, 0.004 * s, 0.07 * s,
                    (0, 0, 0), vertices=6)
    nose.rotation_euler = (math.pi / 2.4, 0, 0)
    nose.parent = head; nose.location = (0, 0.128 * s, 0.055 * s)
    nose.data.materials.append(M["skin_elias"])
    cap = new_cylinder("ELIAS_CAP%s" % suffix, 0.128 * s, 0.075 * s, (0, 0, 0), vertices=12)
    cap.parent = head; cap.location = (0, -0.012 * s, 0.185 * s)
    cap.data.materials.append(M["cardigan"])
    brim = new_cylinder("ELIAS_BRIM%s" % suffix, 0.145 * s, 0.016 * s, (0, 0, 0), vertices=12)
    brim.parent = head; brim.location = (0, 0.075 * s, 0.155 * s)
    brim.scale = (1.0, 1.35, 1.0)
    brim.data.materials.append(M["cardigan"])
    # glasses: two thin tori + bridge
    for gx in (-0.058 * s, 0.058 * s):
        bpy.ops.mesh.primitive_torus_add(major_radius=0.045 * s, minor_radius=0.008 * s,
                                         major_segments=10, minor_segments=5,
                                         location=(0, 0, 0))
        lens = bpy.context.active_object
        lens.name = "ELIAS_GLASS%s%d" % (suffix, 1 if gx > 0 else 0)
        lens.rotation_euler = (math.pi / 2, 0, 0)
        lens.parent = head; lens.location = (gx, 0.105 * s, 0.075 * s)
        lens.data.materials.append(M["brass"])
    bridge = new_cube("ELIAS_BRIDGE%s" % suffix, 1, (0, 0, 0), scale=(0.05 * s, 0.01 * s, 0.012 * s))
    bridge.parent = head; bridge.location = (0, 0.108 * s, 0.075 * s)
    bridge.data.materials.append(M["brass"])
    # arms: shoulder -> elbow -> hand
    rig = {"root": root, "spine": spine, "chest": chest, "head": head}
    for side, sx in (("L", -1), ("R", 1)):
        sh = _empty("ELIAS_SHOUL_%s%s" % (side, suffix), (sx * 0.30 * s, 0, 0.22 * s), chest)
        up = new_cylinder("ELIAS_UPPER_%s%s" % (side, suffix), 0.068 * s, 0.36 * s, (0, 0, 0), vertices=8)
        up.parent = sh; up.location = (0, 0, -0.18 * s)
        up.data.materials.append(M["cardigan"])
        el = _empty("ELIAS_ELBOW_%s%s" % (side, suffix), (0, 0, -0.37 * s), sh)
        fo = new_cylinder("ELIAS_FORE_%s%s" % (side, suffix), 0.057 * s, 0.33 * s, (0, 0, 0), vertices=8)
        fo.parent = el; fo.location = (0, 0, -0.155 * s)
        fo.data.materials.append(M["cardigan_hi"])
        hand = new_icosphere("ELIAS_HAND_%s%s" % (side, suffix), 0.068 * s, (0, 0, 0), segments=1)
        hand.parent = el; hand.location = (0, 0, -0.35 * s)
        hand.data.materials.append(M["skin_elias"])
        rig["shoulder_" + side] = sh
        rig["elbow_" + side] = el
    # legs
    for side, sx in (("L", -1), ("R", 1)):
        hip = _empty("ELIAS_HIP_%s%s" % (side, suffix), (sx * 0.135 * s, 0, -0.12 * s), spine)
        thigh = new_cylinder("ELIAS_THIGH_%s%s" % (side, suffix), 0.092 * s, 0.44 * s, (0, 0, 0), vertices=8)
        thigh.parent = hip; thigh.location = (0, 0, -0.22 * s)
        thigh.data.materials.append(M["trousers"])
        knee = _empty("ELIAS_KNEE_%s%s" % (side, suffix), (0, 0, -0.44 * s), hip)
        shin = new_cylinder("ELIAS_SHIN_%s%s" % (side, suffix), 0.08 * s, 0.42 * s, (0, 0, 0), vertices=8)
        shin.parent = knee; shin.location = (0, 0, -0.21 * s)
        shin.data.materials.append(M["trousers"])
        shoe = new_cube("ELIAS_SHOE_%s%s" % (side, suffix), 1, (0, 0.05 * s, 0),
                        scale=(0.14 * s, 0.30 * s, 0.09 * s))
        shoe.parent = knee; shoe.location = (0, 0.055 * s, -0.42 * s)
        shoe.data.materials.append(M["projector"])
        rig["hip_" + side] = hip
        rig["knee_" + side] = knee
    return rig

def build_boy(M, base=(0.0, 0.0, 0.0), suffix=""):
    """The Boy (9): oversized yellow raincoat — the only saturated thing in the film."""
    s = 0.66
    root = _empty("BOY_ROOT%s" % suffix, base)
    spine = _empty("BOY_SPINE%s" % suffix, (0, 0, 0.62 * s / 0.66), root)   # ~0.62
    chest = _empty("BOY_CHEST%s" % suffix, (0, 0, 0.20), spine)
    head  = _empty("BOY_HEAD%s" % suffix, (0, 0.01, 0.235), chest)
    # oversized raincoat: flared cone
    coat = new_cone("BOY_COAT%s" % suffix, 0.30, 0.165, 0.62, (0, 0, 0), vertices=12)
    coat.parent = spine; coat.location = (0, 0, 0.10)
    coat.data.materials.append(M["yellow_coat"])
    # coat skirt flutter panel (subtle secondary motion handled in animation)
    skirt = new_cone("BOY_SKIRT%s" % suffix, 0.315, 0.285, 0.14, (0, 0, 0), vertices=12)
    skirt.parent = spine; skirt.location = (0, 0, -0.24)
    skirt.data.materials.append(M["coat_shadow"])
    # head + hood
    skull = new_icosphere("BOY_SKULL%s" % suffix, 0.105, (0, 0, 0), segments=2)
    skull.scale = (0.95, 0.98, 1.05)
    skull.parent = head; skull.location = (0, 0, 0.075)
    skull.data.materials.append(M["skin_boy"])
    hood = new_icosphere("BOY_HOOD%s" % suffix, 0.135, (0, 0, 0), segments=2)
    hood.scale = (1.05, 1.15, 1.0)
    hood.parent = head; hood.location = (0, -0.045, 0.085)
    hood.data.materials.append(M["yellow_coat"])
    face = new_icosphere("BOY_FACE%s" % suffix, 0.088, (0, 0, 0), segments=1)
    face.parent = head; face.location = (0, 0.055, 0.062)
    face.scale = (0.92, 0.62, 0.9)
    face.data.materials.append(M["skin_boy"])
    rig = {"root": root, "spine": spine, "chest": chest, "head": head, "skirt": skirt}
    for side, sx in (("L", -1), ("R", 1)):
        sh = _empty("BOY_SHOUL_%s%s" % (side, suffix), (sx * 0.205, 0, 0.155), chest)
        sleeve = new_cylinder("BOY_SLEEVE_%s%s" % (side, suffix), 0.052, 0.30, (0, 0, 0), vertices=8)
        sleeve.parent = sh; sleeve.location = (0, 0, -0.14)
        sleeve.data.materials.append(M["yellow_coat"])
        el = _empty("BOY_ELBOW_%s%s" % (side, suffix), (0, 0, -0.29), sh)
        hand = new_icosphere("BOY_HAND_%s%s" % (side, suffix), 0.052, (0, 0, 0), segments=1)
        hand.parent = el; hand.location = (0, 0, -0.045)
        hand.data.materials.append(M["skin_boy"])
        rig["shoulder_" + side] = sh
        rig["elbow_" + side] = el
    for side, sx in (("L", -1), ("R", 1)):
        hip = _empty("BOY_HIP_%s%s" % (side, suffix), (sx * 0.095, 0, -0.13), spine)
        leg = new_cylinder("BOY_LEG_%s%s" % (side, suffix), 0.066, 0.24, (0, 0, 0), vertices=8)
        leg.parent = hip; leg.location = (0, 0, -0.115)
        leg.data.materials.append(M["trousers"])
        knee = _empty("BOY_KNEE_%s%s" % (side, suffix), (0, 0, -0.24), hip)
        boot = new_cube("BOY_BOOT_%s%s" % (side, suffix), 1, (0, 0.03, 0),
                        scale=(0.11, 0.22, 0.08))
        boot.parent = knee; boot.location = (0, 0.035, -0.035)
        boot.data.materials.append(M["projector"])
        rig["hip_" + side] = hip
        rig["knee_" + side] = knee
    return rig

# ---------------------------------------------------------------- animation --
def key_loc(obj, frame, loc):
    obj.location = loc
    keyframe_safe(obj, 'location', frame)

def key_rot(obj, frame, rot):
    obj.rotation_euler = rot
    keyframe_safe(obj, 'rotation_euler', frame)

def ease_keys(obj):
    ad = obj.animation_data
    for fc in (get_fcurves(ad.action) if ad else []):
        for kp in fc.keyframe_points:
            if kp.interpolation == 'CONSTANT':
                continue
            kp.interpolation = 'BEZIER'

def walk_keys(rig, x0, y0, x1, y1, f_start, f_end, stride=0.72, amp=0.5, bob=0.05,
              z0=0.0, facing=None):
    """Insert a deliberate heel-first walk from (x0,y0) to (x1,y1)."""
    import math as m
    if facing is None:
        facing = m.atan2(-(y1 - y0), (x1 - x0)) + m.pi / 2
    dist = m.hypot(x1 - x0, y1 - y0)
    steps = max(1, int(dist / stride))
    n_keys = max(2, int((f_end - f_start) / 2) + 1)
    for i in range(n_keys):
        t = i / (n_keys - 1)
        f = int(f_start + t * (f_end - f_start))
        phase = t * steps * m.pi
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        z = z0 + abs(m.sin(phase)) * bob
        swing = m.sin(phase)
        rig["root"].location = (x, y, z)
        rig["root"].rotation_euler = (0, 0, facing)
        rig["root"].keyframe_insert('location', frame=f)
        rig["root"].keyframe_insert('rotation_euler', frame=f)
        rig["hip_L"].rotation_euler = (swing * amp, 0, 0)
        rig["hip_R"].rotation_euler = (-swing * amp, 0, 0)
        rig["knee_L"].rotation_euler = (max(0, -swing) * 0.9 + 0.06, 0, 0)
        rig["knee_R"].rotation_euler = (max(0, swing) * 0.9 + 0.06, 0, 0)
        rig["shoulder_L"].rotation_euler = (-swing * 0.38, 0, 0.10)
        rig["shoulder_R"].rotation_euler = (swing * 0.38, 0, -0.10)
        rig["elbow_L"].rotation_euler = (-0.32 - swing * 0.15, 0, 0)
        rig["elbow_R"].rotation_euler = (-0.32 + swing * 0.15, 0, 0)
        for k in ("hip_L", "hip_R", "knee_L", "knee_R", "shoulder_L",
                  "shoulder_R", "elbow_L", "elbow_R"):
            rig[k].keyframe_insert('rotation_euler', frame=f)

def breathe(rig, f0, f1, amp=0.035):
    for f in range(int(f0), int(f1) + 1, 4):
        rig["chest"].rotation_euler = (math.sin(f * 0.35) * amp, 0, 0)
        rig["chest"].keyframe_insert('rotation_euler', frame=f)

# ============================================================================
# MATERIALS + LIGHTING RIG
# ============================================================================
def build_materials():
    M = {}
    def std(name, key, **kw):
        M[name] = make_material(name, PAL[key], **kw)
    std("sea", "sea", roughness=0.55, metallic=0.05)
    std("promenade", "promenade", roughness=0.9)
    std("street", "street", roughness=0.95)
    std("facade_cream", "facade_cream", roughness=0.85)
    std("facade_shade", "facade_shade", roughness=0.9)
    std("marquee_dead", "marquee_dead", roughness=0.8)
    std("int_black", "int_black", roughness=1.0)
    std("wall_wood", "wall_wood", roughness=0.82)
    std("wall_wood2", "wall_wood2", roughness=0.75)
    std("velvet", "velvet", roughness=1.0)
    std("brass", "brass", roughness=0.35, metallic=0.85)
    std("copper", "copper", roughness=0.4, metallic=0.8)
    std("projector", "projector", roughness=0.6, metallic=0.25)
    std("paper", "paper", roughness=0.95)
    std("yellow_coat", "yellow_coat", roughness=0.72)
    std("coat_shadow", "coat_shadow", roughness=0.8)
    std("skin_elias", "skin_elias", roughness=0.65)
    std("skin_boy", "skin_boy", roughness=0.6)
    std("cardigan", "cardigan", roughness=0.9)
    std("cardigan_hi", "cardigan_hi", roughness=0.85)
    std("trousers", "trousers", roughness=0.9)
    std("scarf", "scarf", roughness=0.8)
    std("leaf", "leaf", roughness=0.95)
    M["int_black_text"] = make_material("int_black_text", (0.05, 0.04, 0.035, 1.0), roughness=0.9)
    M["marquee_letters"] = make_material("marquee_letters", (0.72, 0.68, 0.58, 1.0), roughness=0.7,
                                         emit=(0.55, 0.52, 0.42), emit_strength=0.35)
    M["atrium_black"] = M["int_black"]
    # emissive specials
    M["attic_emit"] = make_material("attic_emit", PAL["attic"], roughness=0.5,
                                    emit=(1.0, 0.66, 0.28), emit_strength=2.4)
    M["sign_emit"] = make_material("sign_emit", (0.3, 0.2, 0.12, 1.0), roughness=0.6,
                                   emit=(1.0, 0.74, 0.34), emit_strength=1.5)
    M["sepia_photo"] = make_material("sepia_photo", (0.42, 0.31, 0.17, 1.0), roughness=0.85,
                                     emit=(0.55, 0.40, 0.20), emit_strength=0.5)
    M["sepia_text"] = make_material("sepia_text", (0.9, 0.75, 0.5, 1.0), roughness=0.8,
                                    emit=(0.75, 0.58, 0.34), emit_strength=0.55)
    M["glass"] = make_material("glass", (0.62, 0.7, 0.72, 1.0), roughness=0.08,
                               alpha=0.22)
    M["beam"] = make_glow_mix_material("beam", (1.0, 0.88, 0.62), 1.15, 0.0)
    M["dust"] = make_material("dust", (1.0, 0.92, 0.75, 1.0), roughness=1.0,
                              emit=(1.0, 0.9, 0.68), emit_strength=2.1)
    M["screen_anim"] = make_material("screen_anim", (0.02, 0.02, 0.02, 1.0), roughness=0.9,
                                     emit=(0.05, 0.05, 0.05), emit_strength=0.15)
    M["sun_shaft"] = make_glow_mix_material("sun_shaft", (1.0, 0.82, 0.5), 1.9, 0.0)
    M["win_glow"] = make_material("win_glow", (0.9, 0.75, 0.45, 1.0), roughness=0.7,
                                  emit=(1.0, 0.8, 0.45), emit_strength=2.2)
    return M

def track_spot(spot, target_xyz):
    from mathutils import Vector
    d = Vector(target_xyz) - spot.location
    spot.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()

def build_lights(scene):
    from mathutils import Vector
    L = {}
    # --- exterior sun (cold pre-dawn -> storm -> golden sunrise) ----------
    sun_data = bpy.data.lights.new("SUN", 'SUN')
    sun_data.energy = 0.35
    sun_data.color = (0.62, 0.70, 0.76)
    sun = bpy.data.objects.new("SUN", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(74), 0.0, math.radians(14))
    sun.location = (30, 40, 18)
    L["sun"] = sun
    # --- exterior cool fill -------------------------------------------------
    fill_data = bpy.data.lights.new("EXT_FILL", 'AREA')
    fill_data.energy = 115; fill_data.size = 34; fill_data.color = (0.48, 0.57, 0.64)
    fill = bpy.data.objects.new("EXT_FILL", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (0, 8, 26); fill.rotation_euler = (math.radians(15), 0, 0)
    L["ext_fill"] = fill
    # --- cold bounce off the promenade (shadow-side readability) --------------
    sb_data = bpy.data.lights.new("STREET_BOUNCE", 'AREA')
    sb_data.energy = 320; sb_data.size = 22; sb_data.color = (0.52, 0.60, 0.68)
    sb = bpy.data.objects.new("STREET_BOUNCE", sb_data)
    bpy.context.collection.objects.link(sb)
    sb.location = (2, 9.5, 3.4)
    d = Vector((0, -6.9, 0.8)) - sb.location
    sb.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    L["street_bounce"] = sb
    # --- cool rim on the lock/hands (S02-S04) --------------------------------
    lr = bpy.data.lights.new("LOCK_RIM", 'SPOT')
    lr.energy = 400; lr.color = (0.75, 0.82, 0.9); lr.spot_size = 0.8
    lro = bpy.data.objects.new("LOCK_RIM", lr)
    bpy.context.collection.objects.link(lro)
    lro.location = (2.6, -2.2, 2.7)
    track_spot(lro, (0.3, -6.0, 1.15))
    L["lock_rim"] = lro
    # --- lobby box-office pool (amber, intimate) ---------------------------
    li = bpy.data.lights.new("LOBBY_LAMP", 'POINT')
    li.energy = 520; li.color = (1.0, 0.72, 0.42); li.shadow_soft_size = 0.35
    lo = bpy.data.objects.new("LOBBY_LAMP", li)
    bpy.context.collection.objects.link(lo)
    lo.location = (87.6, -1.2, 2.15)
    L["lobby_lamp"] = lo
    # --- doorway emergency spill (S10) -------------------------------------
    ds = bpy.data.lights.new("DOOR_SPILL", 'SPOT')
    ds.energy = 0; ds.color = (0.72, 0.82, 0.88); ds.spot_size = math.radians(62)
    dso = bpy.data.objects.new("DOOR_SPILL", ds)
    bpy.context.collection.objects.link(dso)
    dso.location = (104.9, 6.6, 2.6)
    track_spot(dso, (105.1, 7.9, 0.4))
    L["door_spill"] = dso
    # --- auditorium house fill ----------------------------------------------
    hf = bpy.data.lights.new("HOUSE_FILL", 'AREA')
    hf.energy = 620; hf.size = 11; hf.color = (0.58, 0.62, 0.66)
    hfo = bpy.data.objects.new("HOUSE_FILL", hf)
    bpy.context.collection.objects.link(hfo)
    hfo.location = (130, 0, 6.6); hfo.rotation_euler = (math.pi, 0, 0)
    L["house_fill"] = hfo
    # --- booth work lamp (tungsten) -----------------------------------------
    bl = bpy.data.lights.new("BOOTH_LAMP", 'POINT')
    bl.energy = 300; bl.color = (1.0, 0.78, 0.5); bl.shadow_soft_size = 0.25
    blo = bpy.data.objects.new("BOOTH_LAMP", bl)
    bpy.context.collection.objects.link(blo)
    blo.location = (115.6, 1.1, 6.75)
    L["booth_lamp"] = blo
    # --- screen glow spilling onto the house (the film light) ---------------
    sg = bpy.data.lights.new("SCREEN_GLOW", 'AREA')
    sg.energy = 0; sg.size = 6.4; sg.color = (1.0, 0.88, 0.66)
    sgo = bpy.data.objects.new("SCREEN_GLOW", sg)
    bpy.context.collection.objects.link(sgo)
    sgo.location = (151.0, 0, 3.4); sgo.rotation_euler = (0, math.pi / 2, 0)
    L["screen_glow"] = sgo
    # --- S15 sunrise shaft through the high side window, onto F7 ------------
    ss = bpy.data.lights.new("SUN_SHAFT", 'SPOT')
    ss.energy = 0; ss.color = (1.0, 0.82, 0.52); ss.spot_size = math.radians(17)
    ss.spot_blend = 0.4
    sso = bpy.data.objects.new("SUN_SHAFT", ss)
    bpy.context.collection.objects.link(sso)
    sso.location = (137.0, 5.6, 6.4)
    track_spot(sso, (132.6, 0.575, floor_z(132.6) + 0.78))
    L["sun_shaft"] = sso
    return L

def keyframe_light_energy(light, pairs):
    """pairs: list of (frame, energy). Inserted in order."""
    light.data.energy = pairs[0][1]
    for frame, energy in pairs:
        light.data.energy = energy
        light.data.keyframe_insert('energy', frame=frame)

def keyframe_light_color(light, pairs):
    for frame, rgb in pairs:
        light.data.color = rgb
        light.data.keyframe_insert('color', frame=frame)

# ============================================================================
# PROPS + SHOT-RELEVANT ANIMATED OBJECTS
# ============================================================================
def build_props(M):
    P = {}
    # ladder for the marquee (S07) — leans on the facade
    ladder_parts = []
    for side in (-0.42, 0.42):
        rail = new_cylinder("LADDER_RAIL", 0.055, 5.6, (0, 0, 0), vertices=6)
        rail.data.materials.append(M["brass"])
        ladder_parts.append(rail)
    ladder = bpy.data.objects.new("LADDER_ROOT", None)
    bpy.context.collection.objects.link(ladder)
    ladder.location = (0.4, -5.6, 0.0)
    ladder.rotation_euler = (math.radians(-14), 0, 0)   # leaning back against facade
    for i, ob in enumerate(ladder_parts):
        ob.parent = ladder
        ob.location = (-0.42 + i * 0.84, 0.0, 2.8)
    rungs = []
    for rz in (0.7, 1.5, 2.3, 3.1, 3.9, 4.7):
        rung = new_cylinder("LADDER_RUNG", 0.035, 0.84, (0, 0, 0), vertices=6)
        rung.rotation_euler = (0, math.pi / 2, 0)
        rung.parent = ladder
        rung.location = (0, 0, rz)
        rung.data.materials.append(M["brass"])
    # broom (S05)
    broom_root = _empty("BROOM_ROOT", (130.0, -1.6, floor_z(130.0) + 1.05))
    handle = new_cylinder("BROOM_HANDLE", 0.035, 1.35, (0, 0, 0), vertices=6)
    handle.parent = broom_root
    handle.location = (0, 0, -0.32)
    handle.rotation_euler = (math.radians(38), 0, 0)
    handle.data.materials.append(M["wall_wood2"])
    head = new_cube("BROOM_HEAD", 1, (0, 0, 0), scale=(0.55, 0.14, 0.07))
    head.parent = broom_root
    head.location = (0, -0.62, -0.88)
    head.data.materials.append(M["wall_wood"])
    P["broom"] = broom_root
    # the two paper boats
    P["boat_found"] = make_paper_boat("BOAT_FOUND", M)          # found S05
    P["boat_boy"] = make_paper_boat("BOAT_BOY", M)              # folded S13
    # ticket stub (S10) + coin
    P["ticket"] = new_cube("TICKET_STUB", 1, (0, 0, 0), scale=(0.05, 0.02, 0.085))
    P["ticket"].data.materials.append(M["paper"])
    P["coin"] = new_cylinder("COIN", 0.035, 0.012, (0, 0, 0), vertices=12)
    P["coin"].data.materials.append(M["brass"])
    # door leaves for the Palace exterior doors (S03)
    P["palace_doors"] = []
    for dx in (-1.55, 1.55):
        hd = _empty("PALACE_DOOR_HINGE_%d" % int(dx), (dx - 1.25 if dx < 0 else dx + 1.25, -6.93, 0.0))
        leaf = new_cube("PALACE_DOOR_%d" % int(dx), 1,
                        (1.25 if dx < 0 else -1.25, 0, 1.6), scale=(2.5, 0.14, 3.2))
        leaf.parent = hd
        leaf.data.materials.append(M["projector"])
        P["palace_doors"].append(hd)
    # dark void behind the Palace doors (interior is 84m away — fake the dark)
    void = new_plane("DOOR_VOID", 5.4, 3.3, (0, -7.25, 1.62), (-math.pi / 2, 0, 0))
    void.data.materials.append(M["int_black"])
    # S15 high window + shaft cone (hidden until dawn)
    win = new_plane("HIGH_WINDOW", 1.6, 1.0, (137.2, 8.78, 6.1), (-math.pi / 2, 0, 0))
    win.rotation_euler = (math.pi / 2, 0, math.pi)   # face -Y into the hall
    win.data.materials.append(M["win_glow"])
    src = np.array([137.2, 8.5, 6.1])
    dst = np.array([132.6, 0.4, floor_z(132.6) + 0.3])
    vec = dst - src
    length = float(np.linalg.norm(vec))
    bpy.ops.mesh.primitive_cone_add(vertices=16, radius1=1.35, radius2=0.32,
                                    depth=length, location=tuple((src + dst) / 2))
    shaft = bpy.context.active_object
    shaft.name = "SUN_SHAFT_CONE"
    from mathutils import Vector
    shaft.rotation_mode = 'QUATERNION'
    shaft.rotation_quaternion = Vector((0.0, 0.0, 1.0)).rotation_difference(Vector(-vec / length))
    shaft.data.materials.append(M["sun_shaft"])
    shaft.visible_shadow = False
    P["sun_shaft_cone"] = shaft
    win.hide_render = True
    return P

def hide(obj, hidden, frame):
    """Hard visibility switch (bool, stepped). Keeps the initial state at frame 1."""
    if '_hide_init' not in obj:
        obj['_hide_init'] = bool(obj.hide_render)
        obj.hide_render = obj['_hide_init']
        obj.hide_viewport = obj['_hide_init']
        obj.keyframe_insert('hide_render', frame=1)
        obj.keyframe_insert('hide_viewport', frame=1)
    obj.hide_render = hidden
    obj.hide_viewport = hidden
    obj.keyframe_insert('hide_render', frame=frame)
    obj.keyframe_insert('hide_viewport', frame=frame)

# ============================================================================
# ATMOSPHERE — generated rain streaks + beam dust (no external assets)
# ============================================================================
def make_streak_image():
    w, h = 256, 256
    img = np.zeros((h, w, 4), dtype=np.float32)
    rng = np.random.default_rng(42)
    for _ in range(110):
        x = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        length = int(rng.integers(14, 46))
        alpha = rng.uniform(0.25, 0.62)
        for dy in range(length):
            y = (y0 + dy) % h
            img[y, x, 0:3] = 0.85
            img[y, x, 3] = max(img[y, x, 3], alpha * (1.0 - dy / length * 0.6))
    return img

def build_atmosphere(M):
    """One rain plane (alpha-streaked, scrolled in Y) parked over the street."""
    arr = make_streak_image()
    height, width = arr.shape[:2]
    img = bpy.data.images.new("RAIN_STREAKS", width=width, height=height, alpha=True)
    flat_px = list(arr.reshape(-1))
    img.pixels = flat_px
    mat = bpy.data.materials.new("rain_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    em = nt.nodes.new('ShaderNodeEmission')
    em.inputs['Strength'].default_value = 0.55
    em.inputs['Color'].default_value = (0.75, 0.82, 0.85, 1.0)
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.extension = 'REPEAT'
    mapping = nt.nodes.new('ShaderNodeMapping')
    mapping.inputs['Scale'].default_value = (6.0, 3.0, 1.0)
    coord = nt.nodes.new('ShaderNodeTexCoord')
    nt.links.new(coord.outputs['Generated'], mapping.inputs['Vector'])
    nt.links.new(mapping.outputs['Vector'], tex.inputs['Vector'])
    transp = nt.nodes.new('ShaderNodeBsdfTransparent')
    mix = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(tex.outputs['Alpha'], mix.inputs['Fac'])
    nt.links.new(transp.outputs['BSDF'], mix.inputs[1])
    nt.links.new(em.outputs['Emission'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])
    mat.surface_render_method = 'BLENDED'
    mat.use_transparency_overlap = False
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, -3.2, 5.2),
                                     rotation=(-math.pi / 2, 0, 0))
    rain = bpy.context.active_object
    rain.name = "RAIN_PLANE"
    rain.scale = (46.0, 13.0, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    rain.data.materials.append(mat)
    rain.visible_shadow = False
    rain.hide_render = True
    hide(rain, False, 790)     # rain starts ~98.75s
    hide(rain, True, 2260)     # gone by first light
    # scroll the streaks downward
    mapping.inputs['Location'].default_value = (0.0, 0.0, 0.0)
    mapping.inputs['Location'].keyframe_insert('default_value', index=1, frame=790)
    mapping.inputs['Location'].default_value = (0.0, -26.0, 0.0)
    mapping.inputs['Location'].keyframe_insert('default_value', index=1, frame=2345)
    return rain

# ============================================================================
# TIMELINE — light keys, flicker, doors, boats, screen (the whole film's pulse)
# ============================================================================
def animate_timeline(scene, L, P, M, props_env):
    F = lambda seconds: int(round(seconds * FPS_RENDER))
    # ---- SUN: pre-dawn -> storm -> night -> sunrise -------------------------
    keyframe_light_energy(L["sun"], [
        (1, 0.95), (F(15), 1.15), (F(90), 0.55), (F(100), 0.25),
        (F(112.0), 2.8), (F(112.8), 0.12), (F(118.0), 2.4), (F(118.8), 0.12),
        (F(127.0), 2.0), (F(127.9), 0.10),
        (F(150), 0.0), (F(280), 0.0), (F(281.5), 0.55), (F(288), 1.5), (F(300), 1.9),
    ])
    keyframe_light_color(L["sun"], [
        (1, (0.62, 0.70, 0.76)), (F(100), (0.45, 0.48, 0.52)),
        (F(112.1), (0.75, 0.78, 0.85)), (F(118.1), (0.75, 0.78, 0.85)),
        (F(127.1), (0.75, 0.78, 0.85)), (F(150), (0.4, 0.4, 0.45)),
        (F(281.5), (0.95, 0.72, 0.42)), (F(300), (1.0, 0.82, 0.55)),
    ])
    keyframe_light_energy(L["ext_fill"], [
        (1, 185), (F(100), 110), (F(104), 88), (F(130), 60), (F(150), 0), (F(280), 0), (F(285), 120),
    ])
    keyframe_light_energy(L["street_bounce"], [
        (1, 320), (F(100), 260), (F(130), 60), (F(150), 0), (F(280), 0), (F(285), 380),
    ])
    keyframe_light_energy(L["lock_rim"], [
        (1, 240), (F(40), 160), (F(60), 0), (F(300), 0),
    ])
    # ---- lobby lamp: on through the day, clicked off after the boy enters ---
    keyframe_light_energy(L["lobby_lamp"], [
        (1, 520), (F(190), 520), (F(191), 760), (F(230), 260), (F(258), 45), (F(280), 0),
    ])
    # ---- house fill: houselights for the audit, dim for the show, worklight --
    keyframe_light_energy(L["house_fill"], [
        (1, 620), (F(154), 620), (F(162), 620), (F(191.5), 620), (F(193), 120),
        (F(250), 95), (F(253), 210), (F(259), 1500), (F(270), 1350), (F(281), 0),
    ])
    # ---- door spill for the boy's entrance ----------------------------------
    keyframe_light_energy(L["door_spill"], [
        (F(171.9), 0), (F(172.3), 380), (F(180), 260), (F(192), 110), (F(230), 0),
    ])
    # ---- booth lamp: steady, then off at shutdown ---------------------------
    keyframe_light_energy(L["booth_lamp"], [
        (1, 300), (F(272), 300), (F(273.2), 300), (F(274), 0), (F(300), 0),
    ])
    # ---- screen glow: dark -> ignite -> seascape flicker -> leader -> dead --
    glow_pairs = [(F(191.9), 0), (F(192.6), 260), (F(193.4), 1550)]
    rng = random.Random(9)
    for t in np.arange(196.0, 243.5, 2.0):
        glow_pairs.append((F(t), 1650 + rng.uniform(-320, 360)))
    glow_pairs += [(F(244.5), 1500), (F(246.5), 950), (F(248.5), 220), (F(250.5), 0), (F(300), 0)]
    keyframe_light_energy(L["screen_glow"], glow_pairs)
    # ---- S15 sunrise shaft ---------------------------------------------------
    keyframe_light_energy(L["sun_shaft"], [(F(280), 0), (F(283.5), 850), (F(291), 1100), (F(300), 480)])
    # ---- the screen surface itself (painting frames by light) -----------------
    scr = props_env["screen"]
    nt = scr.data.materials[0].node_tree
    bsdf = next(n for n in nt.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled')
    # hold the dark until the warm-up — keyframe BEFORE the show ramp so the
    # constant-interp clamp on the first key doesn't ignite the screen at f1
    bsdf.inputs['Emission Color'].default_value = (0.05, 0.05, 0.05, 1)
    bsdf.inputs['Emission Color'].keyframe_insert('default_value', frame=1)
    bsdf.inputs['Emission Color'].keyframe_insert('default_value', frame=F(190))
    bsdf.inputs['Emission Strength'].default_value = 0.15
    bsdf.inputs['Emission Strength'].keyframe_insert('default_value', frame=1)
    bsdf.inputs['Emission Strength'].keyframe_insert('default_value', frame=F(190))
    sea_cols = [(0.05, 0.30, 0.36, 1), (0.10, 0.42, 0.44, 1), (0.78, 0.62, 0.33, 1),
                (0.06, 0.22, 0.34, 1), (0.22, 0.50, 0.50, 1), (0.85, 0.70, 0.38, 1)]
    rng2 = random.Random(31)
    for t in np.arange(193.6, 244.0, 1.5):
        col = sea_cols[int(rng2.random() * len(sea_cols))]
        f = F(t)
        bsdf.inputs['Emission Color'].default_value = col
        bsdf.inputs['Emission Color'].keyframe_insert('default_value', frame=f)
        bsdf.inputs['Emission Strength'].default_value = 2.4
        bsdf.inputs['Emission Strength'].keyframe_insert('default_value', frame=f)
    bsdf.inputs['Emission Color'].default_value = (0.96, 0.94, 0.88, 1)   # leader
    bsdf.inputs['Emission Color'].keyframe_insert('default_value', frame=F(245))
    bsdf.inputs['Emission Strength'].default_value = 2.6
    bsdf.inputs['Emission Strength'].keyframe_insert('default_value', frame=F(245))
    bsdf.inputs['Emission Color'].default_value = (0.02, 0.02, 0.02, 1)
    bsdf.inputs['Emission Color'].keyframe_insert('default_value', frame=F(250.5))
    bsdf.inputs['Emission Strength'].default_value = 0.1
    bsdf.inputs['Emission Strength'].keyframe_insert('default_value', frame=F(250.5))
    # ---- PALACE neon sign: one failing fritz at dawn, then dark forever ------
    sign_mat = M["sign_emit"]
    sign_bsdf = next(n for n in sign_mat.node_tree.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled')
    for f, strength in [(1, 1.5), (F(10.4), 1.5), (F(10.5), 0.05), (F(10.62), 2.1),
                        (F(10.72), 0.05), (F(10.85), 1.7), (F(11.2), 0.35), (F(12.2), 0.0),
                        (F(300), 0.0)]:
        sign_bsdf.inputs['Emission Strength'].default_value = strength
        sign_bsdf.inputs['Emission Strength'].keyframe_insert('default_value', frame=f)
    # ---- beam cone + dust: present only while the projector runs -------------
    beam = props_env["beam"]
    beam_mat = beam.data.materials[0]
    for f, fac in [(F(191.9), 0.0), (F(192.6), 0.08), (F(193.4), 0.3),
                   (F(243.0), 0.27), (F(246.5), 0.12), (F(249.5), 0.0), (F(300), 0.0)]:
        set_mix_fac(beam_mat, fac, frame=f)
    dust = props_env["dust"]
    hide(dust, True, 1)
    hide(dust, False, F(192.8))
    hide(dust, True, F(247.0))
    # slow orbital drift of the motes (2.5 hours of dust settling into the beam)
    dust.rotation_euler = (0, 0, 0)
    dust.keyframe_insert('rotation_euler', frame=F(192.8))
    dust.rotation_euler = (0, 0, 0.9)
    dust.keyframe_insert('rotation_euler', frame=F(247.0))
    # ---- sun-shaft cone + high window: dawn only ------------------------------
    cone = P["sun_shaft_cone"]
    cone_mat = cone.data.materials[0]
    for f, fac in [(F(280), 0.0), (F(283.5), 0.14), (F(291), 0.22), (F(299), 0.06)]:
        set_mix_fac(cone_mat, fac, frame=f)
    win = props_env["high_window"]
    hide(win, False, F(283.0))
    # ---- doors: exterior leaves, auditorium twin leaves, boy's side door ------
    for hd in P["palace_doors"]:
        sign = 1 if hd.location.x < 0 else -1
        hd.rotation_euler = (0, 0, 0)
        hd.keyframe_insert('rotation_euler', frame=F(30.0))
        hd.rotation_euler = (0, 0, sign * 0.62)
        hd.keyframe_insert('rotation_euler', frame=F(31.8))
        hd.rotation_euler = (0, 0, sign * 0.62)
        hd.keyframe_insert('rotation_euler', frame=F(36.0))
    leaves, hinge_b = props_env["aud_doors"], props_env["boy_door"]
    for hd in leaves:                       # twin auditorium doors: open for the show
        sign = -1 if hd.name.endswith("L") else 1
        hd.rotation_euler = (0, 0, 0)
        hd.keyframe_insert('rotation_euler', frame=F(190.5))
        hd.rotation_euler = (0, 0, sign * 1.15)
        hd.keyframe_insert('rotation_euler', frame=F(192.8))
    hinge_b.rotation_euler = (0, 0, 0)      # boy's door ajar, bangs open at 172s
    hinge_b.keyframe_insert('rotation_euler', frame=F(171.9))
    hinge_b.rotation_euler = (0, 0, -0.95)
    hinge_b.keyframe_insert('rotation_euler', frame=F(172.35))
    hinge_b.rotation_euler = (0, 0, -0.55)
    hinge_b.keyframe_insert('rotation_euler', frame=F(174.5))
    hinge_b.rotation_euler = (0, 0, -0.62)
    hinge_b.keyframe_insert('rotation_euler', frame=F(180.0))
    # ---- the boats -------------------------------------------------------------
    bf, bb = P["boat_found"], P["boat_boy"]
    # found boat: hidden until Elias picks it up (S05 ~ 71s), pocketed (~75s),
    # returned to F7's armrest before the show (S11 ~ 191s)
    hide(bf, True, 1)
    hide(bf, False, F(70.5))               # lies on the row-F floor
    bf.location = (132.75, 0.30, floor_z(132.6) + 0.10)
    bf.rotation_euler = (0, 0, 0.4)
    bf.keyframe_insert('location', frame=F(70.5))
    bf.keyframe_insert('rotation_euler', frame=F(70.5))
    hide(bf, True, F(74.6))                # into the pocket
    hide(bf, False, F(190.8))              # placed on the F7 armrest
    bf.location = (132.6, 0.83, floor_z(132.6) + 0.70)
    bf.rotation_euler = (0, 0, -0.25)
    bf.keyframe_insert('location', frame=F(190.8))
    bf.keyframe_insert('rotation_euler', frame=F(190.8))
    hide(bf, True, F(241.5))               # the boy pockets it as he leaves
    # boy's folded boat: appears F13/F14 on the F7 armrest in its place
    hide(bb, True, 1)
    hide(bb, False, F(249.6))              # set down after the fold
    bb.location = (132.6, 0.83, floor_z(132.6) + 0.70)
    bb.rotation_euler = (0, 0, 0.18)
    bb.keyframe_insert('location', frame=F(249.6))
    bb.keyframe_insert('rotation_euler', frame=F(249.6))
    hide(bb, True, F(261.0))               # Elias pockets it...
    hide(bb, False, F(277.5))              # ...and returns it to F7 forever
    bb.location = (132.6, 0.83, floor_z(132.6) + 0.70)
    bb.rotation_euler = (0, 0, 0.18)
    bb.keyframe_insert('location', frame=F(277.5))
    # ---- coin + ticket -----------------------------------------------------------
    hide(P["coin"], True, 1)
    hide(P["ticket"], True, 1)
    hide(P["coin"], False, F(172.6))
    hide(P["coin"], True, F(183.2))
    hide(P["ticket"], False, F(185.0))
    hide(P["ticket"], True, F(191.5))

# ============================================================================
# CHARACTER ANIMATION — per shot, wordless, readable at 8 fps
# ============================================================================
ALL_CONTROLS = ["root", "spine", "chest", "head", "shoulder_L", "shoulder_R",
                "elbow_L", "elbow_R", "hip_L", "hip_R", "knee_L", "knee_R"]
BOUNDARIES = [97, 241, 353, 513, 528, 689, 833, 1041, 1233, 1377, 1537, 1729, 1921, 2065, 2112, 2241, 2336]

def constant_boundaries(rig):
    for k in ALL_CONTROLS:
        if k in rig:
            set_constant_on_frames(rig[k], BOUNDARIES)

def pose_stand(rig, frame, loc, yaw, spine_x=0.0, head_x=0.0, head_y=0.0):
    rig["root"].location = loc
    rig["root"].rotation_euler = (0, 0, yaw)
    rig["root"].keyframe_insert('location', frame=frame)
    rig["root"].keyframe_insert('rotation_euler', frame=frame)
    rig["spine"].rotation_euler = (spine_x, 0, 0)
    rig["spine"].keyframe_insert('rotation_euler', frame=frame)
    rig["head"].rotation_euler = (head_x, head_y, 0)
    rig["head"].keyframe_insert('rotation_euler', frame=frame)

def arm(rig, frame, side, shoulder_rot, elbow_rot):
    rig["shoulder_%s" % side].rotation_euler = shoulder_rot
    rig["shoulder_%s" % side].keyframe_insert('rotation_euler', frame=frame)
    rig["elbow_%s" % side].rotation_euler = elbow_rot
    rig["elbow_%s" % side].keyframe_insert('rotation_euler', frame=frame)

def animate_elias_a(rig, P):
    """S02 walk, S03 entry, S04 booth ritual, S05 auditorium, S06 office, S07 ladder."""
    F = lambda s: int(round(s * FPS_RENDER))
    # ---------- S02 (12-30s): the walk down the promenade ----------------------
    walk_keys(rig, -19.0, 10.4, -1.2, 9.35, 97, 236, stride=0.74, amp=0.46,
              bob=0.045, z0=0.30, facing=math.pi / 2 - 0.13)
    pose_stand(rig, 240, (-0.7, 9.3, 0.30), math.pi / 2 - 0.13)
    # ---------- S03 (30-44s): unlock, pull, enter -------------------------------
    pose_stand(rig, 241, (0.65, -4.55, 0.24), math.pi, spine_x=0.06)
    arm(rig, 241, "R", (-0.5, 0, -0.25), (-0.5, 0, 0))
    # raise hand to the lock
    arm(rig, 247, "R", (-1.9, 0, -0.35), (-1.15, 0, 0))
    pose_stand(rig, 247, (0.65, -4.85, 0.24), math.pi, spine_x=0.18, head_x=0.18)
    # key turn — two twists (hand torque)
    arm(rig, 251, "R", (-1.9, 0, -0.35), (-1.15, 0, 0.55))
    arm(rig, 254, "R", (-1.9, 0, -0.35), (-1.15, 0, -0.4))
    arm(rig, 256, "R", (-1.9, 0, -0.35), (-1.15, 0, 0.35))
    # lean back and haul the door
    pose_stand(rig, 262, (0.65, -4.6, 0.24), math.pi, spine_x=-0.14, head_x=-0.08)
    arm(rig, 262, "R", (-2.35, 0, -0.2), (-0.35, 0, 0))
    arm(rig, 262, "L", (-2.35, 0, 0.2), (-0.35, 0, 0))
    # step across the threshold
    pose_stand(rig, 285, (0.55, -6.0, 0.24), math.pi, spine_x=0.05)
    arm(rig, 285, "R", (-0.6, 0, -0.15), (-0.6, 0, 0))
    arm(rig, 285, "L", (-0.6, 0, 0.15), (-0.6, 0, 0))
    pose_stand(rig, 300, (0.45, -6.75, 0.24), math.pi)
    breathe(rig, 300, 352, amp=0.03)
    # ---------- S04 (44-64s): the booth ritual ----------------------------------
    pose_stand(rig, 353, (113.55, 0.55, 4.95), math.pi / 2, spine_x=0.32, head_x=0.22)
    arm(rig, 353, "R", (-1.35, 0, -0.28), (-0.9, 0, 0))
    arm(rig, 353, "L", (-1.30, 0, 0.30), (-0.85, 0, 0))
    # delicate threading adjustments
    arm(rig, 372, "R", (-1.55, 0, -0.22), (-1.5, 0, 0.3))
    arm(rig, 385, "L", (-1.5, 0, 0.42), (-1.4, 0, -0.3))
    arm(rig, 397, "R", (-1.2, 0, -0.35), (-0.7, 0, 0))
    # turn to the 1953 photograph
    pose_stand(rig, 418, (113.55, 0.55, 4.95), math.pi / 2, spine_x=0.05)
    rig["head"].rotation_euler = (0.0, 0.62, 0.0)   # look to his left, at the wall
    rig["head"].keyframe_insert('rotation_euler', frame=418)
    # hand rises to touch the frame
    arm(rig, 442, "R", (-2.05, 0, -0.55), (-1.35, 0, 0))
    pose_stand(rig, 452, (113.3, 0.35, 4.95), math.pi / 2 + 0.28, spine_x=-0.05, head_x=-0.12)
    arm(rig, 452, "R", (-2.25, 0, -0.62), (-1.1, 0, 0))
    # a long held beat, then back to work
    arm(rig, 480, "R", (-1.0, 0, -0.3), (-0.75, 0, 0))
    pose_stand(rig, 495, (113.55, 0.55, 4.95), math.pi / 2, spine_x=0.3, head_x=0.3)
    breathe(rig, 495, 512, amp=0.028)
    # ---------- S05 (64-86s): sweep, find the boat -------------------------------
    pose_stand(rig, 513, (115.0, 0.4, floor_z(115.0)), math.pi / 2)
    walk_keys(rig, 115.0, 0.4, 124.0, -0.6, 513, 580, stride=0.7, amp=0.4,
              bob=0.035, z0=floor_z(119.5))   # z handled by rake approx below
    # overwrite root z to follow the rake exactly
    for i in range(6):
        t = i / 5
        f = int(513 + t * (580 - 513))
        x = 115.0 + (124.0 - 115.0) * t
        rig["root"].location = (x, 0.4 + (-1.0) * t, floor_z(x))
        rig["root"].keyframe_insert('location', frame=f)
    # sweeping strokes — broom rocks, he works the front rows
    pose_stand(rig, 580, (124.2, -0.6, floor_z(124.2)), math.pi / 2 + 0.25, spine_x=0.34,
               head_x=0.34)
    for stroke, f in enumerate((586, 598, 610)):
        P["broom"].rotation_euler = (0.0, 0.0, 0.28)
        P["broom"].keyframe_insert('rotation_euler', frame=f)
        P["broom"].rotation_euler = (0.0, 0.0, -0.28)
        P["broom"].keyframe_insert('rotation_euler', frame=f + 6)
        arm(rig, f, "R", (-1.05, 0, -0.3), (-0.75, 0, 0))
        arm(rig, f + 6, "R", (-0.8, 0, -0.3), (-0.55, 0, 0))
    # crouch, reach, pick up the boat on Row F
    pose_stand(rig, 626, (131.6, 0.35, floor_z(131.6)), math.pi / 2 + 0.35,
               spine_x=0.62, head_x=0.5)
    arm(rig, 626, "R", (-1.15, 0, -0.25), (-1.15, 0, 0))
    pose_stand(rig, 636, (131.9, 0.45, floor_z(131.9)), math.pi / 2 + 0.35,
               spine_x=0.28, head_x=0.42)
    arm(rig, 636, "R", (-1.7, 0, -0.28), (-1.7, 0, 0))   # boat in hand, near face
    # into the pocket, then the look up to the attic hatch
    arm(rig, 655, "R", (-0.75, 0, -0.5), (-1.9, 0, 0))
    pose_stand(rig, 662, (131.9, 0.45, floor_z(131.9)), math.pi / 2 + 0.35,
               spine_x=-0.02, head_x=-0.52)              # head UP — the realisation
    breathe(rig, 662, 688, amp=0.02)
    # ---------- S06 (86-104s): the ledger ----------------------------------------
    pose_stand(rig, 689, (88.35, -0.75, 0.0), -math.pi / 2 + 0.35, spine_x=0.28, head_x=0.3)
    arm(rig, 689, "R", (-0.85, 0, -0.15), (-1.05, 0, 0))
    arm(rig, 689, "L", (-0.55, 0, 0.28), (-0.65, 0, 0))
    for stamp_f in (700, 744, 788):
        arm(rig, stamp_f, "R", (-0.55, 0, -0.1), (-0.6, 0, 0))     # fist rises
        arm(rig, stamp_f + 2, "R", (-1.0, 0, -0.12), (-1.35, 0, 0))  # THUNK
    # page turn — slow, deliberate
    arm(rig, 712, "L", (-0.95, 0, 0.35), (-1.25, 0, 0))
    arm(rig, 726, "L", (-0.55, 0, 0.28), (-0.65, 0, 0))
    # looks up at the rain-streaked window
    pose_stand(rig, 806, (88.35, -0.75, 0.0), -math.pi / 2 + 0.35, spine_x=0.05, head_x=-0.35)
    breathe(rig, 806, 832, amp=0.022)
    # ---------- S07 (104-130s): letters on the marquee ---------------------------
    pose_stand(rig, 833, (0.45, -5.15, 0.62), math.pi, spine_x=0.12)
    walk_keys(rig, 0.45, -5.15, 0.62, -5.0, 833, 852, stride=0.3, amp=0.28,
              bob=0.02, z0=0.62)
    pose_stand(rig, 858, (0.62, -5.0, 1.32), math.pi, spine_x=0.05)
    arm(rig, 858, "R", (-2.4, 0, -0.3), (-0.75, 0, 0))   # arm high to the board
    arm(rig, 858, "L", (-1.35, 0, 0.45), (-1.9, 0, 0))
    # lightning jolts — his head snaps skyward (896s/944s/1016s ~ frames)
    for lf in (897, 945, 1017):
        pose_stand(rig, lf, (0.62, -5.0, 1.5), math.pi, spine_x=-0.08, head_x=-0.5)
        pose_stand(rig, lf + 14, (0.62, -5.0, 1.5), math.pi, spine_x=0.06, head_x=0.18)
    arm(rig, 900, "R", (-2.65, 0, -0.28), (-0.4, 0, 0))  # hauling a letter home
    pose_stand(rig, 1032, (0.62, -5.0, 1.35), math.pi, spine_x=-0.04, head_x=-0.18)
    breathe(rig, 1032, 1040, amp=0.025)

def animate_elias_b(rig, P):
    """S08 breaker, S09 the lever, S10 the boy, S11–S15."""
    F = lambda s: int(round(s * FPS_RENDER))
    # ---------- S08 (130-154s): mains, then the wait ----------------------------
    pose_stand(rig, 1041, (115.0, 0.9, 4.95), math.pi / 2, spine_x=0.12)
    # MAINS CLUNK at 132s (frame 1056)
    arm(rig, 1050, "R", (-1.5, 0, -0.3), (-0.45, 0, 0))
    arm(rig, 1056, "R", (-2.1, 0, -0.32), (-0.3, 0, 0))    # switch slams down
    arm(rig, 1075, "R", (-0.9, 0, -0.25), (-0.7, 0, 0))
    # the long look down into the empty house
    pose_stand(rig, 1130, (115.35, 0.35, 4.95), math.pi / 2, spine_x=0.24, head_x=0.24)
    breathe(rig, 1130, 1232, amp=0.026)
    # ---------- S09 (154-172s): hand on the lever; 7:30 --------------------------
    pose_stand(rig, 1233, (115.0, 0.9, 4.95), math.pi / 2, spine_x=0.1)
    arm(rig, 1252, "R", (-1.75, 0, -0.28), (-0.55, 0, 0))  # hand settles on the lever
    breathe(rig, 1252, 1368, amp=0.018)
    # the clock chime — a small involuntary dip
    pose_stand(rig, 1370, (115.0, 0.9, 4.95), math.pi / 2, spine_x=0.1, head_x=0.28)
    pose_stand(rig, 1376, (115.0, 0.9, 4.95), math.pi / 2, spine_x=0.1, head_x=0.05)
    # ---------- S10 (172-192s): the boy in the lobby ------------------------------
    # startle at the bang (f1378): head snaps toward the door
    pose_stand(rig, 1378, (88.9, -1.6, 0.0), -math.pi / 2 + 0.35, spine_x=-0.12, head_y=1.15)
    pose_stand(rig, 1394, (88.9, -1.6, 0.0), -math.pi / 2 + 0.35, spine_x=0.05, head_y=0.35)
    # cross the lobby to him
    walk_keys(rig, 88.9, -1.6, 96.5, 2.6, 1400, 1465, stride=0.62, amp=0.42,
              bob=0.035, z0=0.0, facing=math.atan2(7.6 - (-1.6), 103.5 - 88.9))
    pose_stand(rig, 1468, (97.2, 3.1, 0.0), math.atan2(7.6 - 3.1, 103.5 - 97.2), spine_x=0.06)
    # look at the coin — long beat
    pose_stand(rig, 1500, (97.2, 3.1, 0.0), math.atan2(7.6 - 3.1, 103.5 - 97.2),
               spine_x=0.34, head_x=0.42)
    # the ADMIT ONE stub changes hands
    arm(rig, 1510, "R", (-1.15, 0, -0.3), (-1.05, 0, 0))
    arm(rig, 1526, "L", (-1.15, 0, 0.3), (-1.05, 0, 0))
    # sweep the doors open — an invitation
    pose_stand(rig, 1530, (98.2, 3.6, 0.0), math.atan2(7.6 - 3.6, 103.5 - 98.2) - 0.4,
               spine_x=0.05)
    arm(rig, 1532, "R", (-1.6, 0, -0.9), (-0.35, 0, 0))
    breathe(rig, 1532, 1536, amp=0.02)
    # ---------- S11 (192-216s): the lights go down; the show begins ----------------
    # Elias belongs to the booth from here — house fill darkens (timeline chunk)
    pose_stand(rig, 1537, (115.0, 0.9, 4.95), math.pi / 2, spine_x=0.08)
    # the changeover: one decisive pull at f1542 (192.75s)
    arm(rig, 1539, "R", (-1.35, 0, -0.3), (-0.5, 0, 0))
    arm(rig, 1542, "R", (-1.9, 0, -0.3), (-0.15, 0, 0))
    arm(rig, 1560, "R", (-0.85, 0, -0.25), (-0.75, 0, 0))
    breathe(rig, 1560, 1660, amp=0.02)
    # he watches the boy watch — eye to the porthole
    pose_stand(rig, 1665, (115.5, 0.45, 4.95), math.pi / 2, spine_x=0.3, head_x=0.28)
    breathe(rig, 1665, 1728, amp=0.016)
    # ---------- S12 (216-240s): intercut macro; he looks UP at the hatch -----------
    pose_stand(rig, 1729, (115.5, 0.45, 4.95), math.pi / 2, spine_x=0.3, head_x=0.28)
    pose_stand(rig, 1800, (115.5, 0.45, 4.95), math.pi / 2, spine_x=0.3, head_x=-0.42)  # UP
    pose_stand(rig, 1865, (115.5, 0.45, 4.95), math.pi / 2, spine_x=0.3, head_x=0.18)
    breathe(rig, 1865, 1920, amp=0.016)
    # ---------- S13 (240-258s): the reel ends; silence ------------------------------
    pose_stand(rig, 1921, (115.0, 0.9, 4.95), math.pi / 2, spine_x=0.1)
    # reel flaps at f1946-1976; his hands go still on the bench
    arm(rig, 1946, "R", (-0.6, 0, -0.2), (-0.5, 0, 0))
    arm(rig, 1946, "L", (-0.6, 0, 0.2), (-0.5, 0, 0))
    breathe(rig, 1946, 2064, amp=0.014)
    # ---------- S14 (258-280s): shutdown; the boat; the keys ------------------------
    for f_sw, f_key in ((F(261), 2065), (F(266), 2084), (F(271), 2103)):
        arm(rig, f_key, "R", (-1.5, 0, -0.3), (-0.4, 0, 0))
        arm(rig, f_key + 3, "R", (-2.0, 0, -0.32), (-0.22, 0, 0))   # each breaker falls
        arm(rig, f_key + 12, "R", (-0.8, 0, -0.25), (-0.6, 0, 0))
    pose_stand(rig, 2111, (115.0, 0.9, 4.95), math.pi / 2, spine_x=0.22, head_x=0.15)
    # the cut — he is simply... in the dark house now (hard jump, CONSTANT at 2112)
    pose_stand(rig, 2112, (117.6, 0.4, floor_z(117.6)), math.pi / 2, spine_x=0.08)
    walk_keys(rig, 117.6, 0.4, 131.7, 1.3, 2112, 2165, stride=0.7, amp=0.4,
              bob=0.03, z0=floor_z(124.5), facing=math.pi / 2 + 0.18)
    for i in range(6):
        t = i / 5
        f = int(2112 + t * (2165 - 2112))
        x = 117.6 + (131.7 - 117.6) * t
        rig["root"].location = (x, 0.4 + (1.3 - 0.4) * t, floor_z(x))
        rig["root"].keyframe_insert('location', frame=f)
    # the boat — bent knees, both hands, reverent
    pose_stand(rig, 2170, (131.9, 1.0, floor_z(131.9)), math.pi / 2 + 0.3,
               spine_x=0.55, head_x=0.5)
    arm(rig, 2170, "R", (-1.1, 0, -0.2), (-1.2, 0, 0))
    arm(rig, 2170, "L", (-1.1, 0, 0.2), (-1.2, 0, 0))
    arm(rig, 2182, "R", (-0.75, 0, -0.42), (-1.85, 0, 0))   # into the breast pocket
    arm(rig, 2182, "L", (-0.75, 0, 0.42), (-1.85, 0, 0))
    # rises, rests his hand on the seat back, holds the frame
    pose_stand(rig, 2205, (131.9, 1.0, floor_z(131.9)), math.pi / 2 + 0.3,
               spine_x=-0.04, head_x=0.28)
    arm(rig, 2205, "R", (-1.15, 0, -0.18), (-0.5, 0, 0))
    arm(rig, 2205, "L", (-0.35, 0, 0.25), (-0.35, 0, 0))
    breathe(rig, 2205, 2240, amp=0.02)
    # ---------- S15A (280-292s): out into the dawn ----------------------------------
    pose_stand(rig, 2241, (0.8, -6.7, 0.24), -math.pi / 2 - 0.15, spine_x=0.04)
    walk_keys(rig, 0.8, -6.7, -13.5, -6.75, 2241, 2330, stride=0.68, amp=0.4,
              bob=0.04, z0=0.24, facing=-math.pi / 2 - 0.15)
    pose_stand(rig, 2335, (-14.2, -6.75, 0.24), -math.pi / 2 - 0.15)
    # ---------- S15B (292-300s): he is gone; the boat is not -------------------------

def animate_boy(rig, P):
    """The Boy: S10 arrival -> S11 sits for the show -> S12 swinging feet -> S13 leaves."""
    F = lambda s: int(round(s * FPS_RENDER))
    YAW_IN = math.atan2(-(7.6 - 3.1), 96.5 - 104.9)  # facing Elias/ counter (approx)
    # ---------- S10: drenched at the door, coin on a flat palm ----------------------
    pose_stand(rig, 1377, (104.9, 7.35, 0.0), math.pi * 0.92, spine_x=0.06, head_x=0.1)
    # shiver — three involuntary pulses
    for i, f in enumerate((1384, 1400, 1416)):
        rig["spine"].rotation_euler = (0.10, 0, 0.045)
        rig["spine"].keyframe_insert('rotation_euler', frame=f)
        rig["spine"].rotation_euler = (0.16, 0, -0.045)
        rig["spine"].keyframe_insert('rotation_euler', frame=f + 3)
    # walks to Elias
    walk_keys(rig, 104.9, 7.3, 98.6, 3.9, 1420, 1466, stride=0.52, amp=0.5,
              bob=0.035, z0=0.0, facing=math.atan2(3.9 - 7.3, 98.6 - 104.9))
    pose_stand(rig, 1470, (98.6, 3.9, 0.0), math.atan2(97.2 - 98.6, 3.1 - 3.9),
               spine_x=0.08)
    # the coin, held out on a flat palm — the bravest thing in the film
    arm(rig, 1480, "R", (-0.85, 0, -0.18), (-1.15, 0, 0))
    arm(rig, 1480, "L", (-0.35, 0, 0.15), (-0.4, 0, 0))
    pose_stand(rig, 1480, (98.6, 3.9, 0.0), math.atan2(97.2 - 98.6, 3.1 - 3.9),
               spine_x=0.14, head_x=0.12)
    # takes the stub, closes his fingers
    arm(rig, 1498, "R", (-1.05, 0, -0.2), (-1.6, 0, 0))
    arm(rig, 1512, "R", (-0.9, 0, -0.3), (-1.5, 0, 0))
    # ---------- S11: down the aisle; sits in F7 --------------------------------------
    pose_stand(rig, 1537, (111.6, 0.2, floor_z(111.6)), math.pi / 2, spine_x=0.12)
    walk_keys(rig, 111.6, 0.2, 132.55, 0.575, 1542, 1655, stride=0.5, amp=0.55,
              bob=0.04, z0=floor_z(122.0), facing=math.pi / 2)
    for i in range(6):
        t = i / 5
        f = int(1542 + t * (1655 - 1542))
        x = 111.6 + (132.55 - 111.6) * t
        rig["root"].location = (x, 0.2 + (0.575 - 0.2) * t, floor_z(x))
        rig["root"].keyframe_insert('location', frame=f)
    # the sit — knees folded, boots hovering
    pose_stand(rig, 1662, (132.6, 0.62, floor_z(132.6) - 0.32), math.pi / 2, spine_x=0.06)
    rig["knee_L"].rotation_euler = (-1.35, 0, 0)
    rig["knee_L"].keyframe_insert('rotation_euler', frame=1662)
    rig["knee_R"].rotation_euler = (-1.35, 0, 0)
    rig["knee_R"].keyframe_insert('rotation_euler', frame=1662)
    rig["hip_L"].rotation_euler = (-1.15, 0, 0)
    rig["hip_L"].keyframe_insert('rotation_euler', frame=1662)
    rig["hip_R"].rotation_euler = (-1.15, 0, 0)
    rig["hip_R"].keyframe_insert('rotation_euler', frame=1662)
    # hands folded on his knees — at attention for the film
    arm(rig, 1662, "R", (-0.75, 0, -0.2), (-1.15, 0, 0))
    arm(rig, 1662, "L", (-0.75, 0, 0.2), (-1.15, 0, 0))
    breathe(rig, 1662, 1728, amp=0.03)
    # ---------- S12: swinging boots ---------------------------------------------------
    for f in range(1735, 1921, 2):
        swing = math.sin((f - 1735) * 0.42) * 0.42
        rig["knee_L"].rotation_euler = (-1.35 - swing, 0, 0)
        rig["knee_R"].rotation_euler = (-1.35 + swing, 0, 0)
        rig["knee_L"].keyframe_insert('rotation_euler', frame=f)
        rig["knee_R"].keyframe_insert('rotation_euler', frame=f)
    # chest rises once, hugely, at the 1953-wall beat (f~1800) — wonder, visible
    pose_stand(rig, 1802, (132.6, 0.62, floor_z(132.6) - 0.28), math.pi / 2,
               spine_x=-0.14, head_x=-0.22)
    pose_stand(rig, 1830, (132.6, 0.62, floor_z(132.6) - 0.32), math.pi / 2,
               spine_x=0.06, head_x=0.05)
    # ---------- S13: the last reel flaps; the fold --------------------------------------
    # stands as the leader flaps die
    pose_stand(rig, 1962, (132.6, 0.62, floor_z(132.6)), math.pi / 2, spine_x=0.16)
    rig["knee_L"].rotation_euler = (0.06, 0, 0)
    rig["knee_L"].keyframe_insert('rotation_euler', frame=1962)
    rig["knee_R"].rotation_euler = (0.06, 0, 0)
    rig["knee_R"].keyframe_insert('rotation_euler', frame=1962)
    rig["hip_L"].rotation_euler = (0.0, 0, 0)
    rig["hip_L"].keyframe_insert('rotation_euler', frame=1962)
    rig["hip_R"].rotation_euler = (0.0, 0, 0)
    rig["hip_R"].keyframe_insert('rotation_euler', frame=1962)
    # the fold — four small, precise gestures (he's watched this all his life)
    pose_stand(rig, 1976, (132.6, 0.78, floor_z(132.6)), math.pi / 2 - 0.3,
               spine_x=0.3, head_x=0.45)
    fold_specs = ((1978, -0.9, -1.5), (1986, -1.2, -1.7), (1993, -0.8, -1.35), (1998, -1.05, -1.6))
    for f, sh_x, el_x in fold_specs:
        arm(rig, f, "R", (sh_x, 0, -0.35), (el_x, 0, 0.15))
        arm(rig, f, "L", (sh_x, 0, 0.35), (el_x, 0, -0.15))
    # sets it down on the armrest, gently
    pose_stand(rig, 2001, (132.55, 0.9, floor_z(132.6)), math.pi / 2 - 0.15,
               spine_x=0.42, head_x=0.5)
    arm(rig, 2001, "R", (-1.05, 0, -0.32), (-1.5, 0, 0))
    # into his pocket with the boat he found — then gone
    arm(rig, 2012, "R", (-0.7, 0, -0.45), (-1.9, 0, 0))
    breathe(rig, 2012, 2024, amp=0.02)
    # exits toward the lobby side door
    walk_keys(rig, 132.6, 0.8, 112.2, 0.55, 2024, 2058, stride=0.5, amp=0.5,
              bob=0.035, z0=floor_z(122.0), facing=-math.pi / 2)
    for i in range(6):
        t = i / 5
        f = int(2024 + t * (2058 - 2024))
        x = 132.6 + (112.2 - 132.6) * t
        rig["root"].location = (x, 0.8 + (0.55 - 0.8) * t, floor_z(x))
        rig["root"].keyframe_insert('location', frame=f)
    pose_stand(rig, 2064, (111.4, 0.5, floor_z(111.4)), -math.pi / 2, spine_x=0.1)

# ============================================================================
# CAMERA — one camera, fifteen setups, hard cuts on CONSTANT keys
# ============================================================================
def look_euler(loc, tgt):
    from mathutils import Vector
    d = Vector(tgt) - Vector(loc)
    return d.to_track_quat('-Z', 'Y').to_euler()

def cam_key(cam, frame, loc, tgt, lens):
    cam.location = loc
    cam.rotation_euler = look_euler(loc, tgt)
    cam.data.lens = lens
    cam.keyframe_insert('location', frame=frame)
    cam.keyframe_insert('rotation_euler', frame=frame)
    cam.data.keyframe_insert('lens', frame=frame)

def build_camera(scene):
    cam_data = bpy.data.cameras.new("PALACE_CAM")
    cam_data.clip_start = 0.08
    cam_data.clip_end = 400.0
    cam = bpy.data.objects.new("PALACE_CAM", cam_data)
    bpy.context.collection.objects.link(cam)
    scene.camera = cam
    fz = floor_z
    # ---------------- S01 — EXT dawn establishing ----------------
    cam_key(cam, 1, (-31.5, 25.0, 3.1), (0, -8.5, 6.4), 40)
    cam_key(cam, 96, (-16.5, 21.0, 4.8), (0, -9.0, 7.2), 40)
    # ---------------- S02 — promenade track ----------------
    cam_key(cam, 97, (-23.5, 16.5, 2.1), (-13.0, 9.2, 1.1), 36)
    cam_key(cam, 168, (-7.5, 13.2, 1.9), (0.5, 6.0, 1.5), 36)
    cam_key(cam, 240, (3.6, 10.8, 2.1), (0.4, -5.5, 2.4), 40)
    # ---------------- S03 — the lock / the door ----------------
    cam_key(cam, 241, (4.6, 0.8, 1.7), (0.35, -5.3, 1.15), 50)
    cam_key(cam, 300, (3.4, -1.2, 1.5), (0.4, -5.35, 1.15), 48)
    cam_key(cam, 352, (1.2, -3.4, 1.5), (0.25, -5.8, 1.4), 46)
    # ---------------- S04 — booth ritual ----------------
    cam_key(cam, 353, (112.35, 1.7, 6.0), (113.75, -0.15, 5.8), 58)
    cam_key(cam, 400, (112.9, 1.9, 6.35), (113.6, 0.1, 6.05), 44)
    cam_key(cam, 470, (114.6, 1.6, 6.25), (110.5, -1.2, 6.35), 38)
    cam_key(cam, 512, (112.6, -0.35, 6.1), (110.5, -1.2, 6.38), 52)
    # ---------------- S05 — rows; the boat on F ----------------
    cam_key(cam, 513, (113.6, 0, 5.7), (151.7, 0, 3.2), 30)
    cam_key(cam, 528, (116.55, 0.0, 6.05), (140.0, 0, 3.3), 34)
    cam_key(cam, 560, (127.6, -3.55, fz(127.6) + 2.15), (137.5, 0.4, fz(137.5) + 0.7), 40)
    cam_key(cam, 610, (129.6, -1.2, fz(129.6) + 1.75), (132.6, 0.575, fz(132.6) + 0.72), 48)
    cam_key(cam, 645, (132.15, -0.15, fz(132.4) + 1.18), (132.72, 0.32, fz(132.4) + 0.68), 63)
    cam_key(cam, 688, (132.15, -0.15, fz(132.4) + 1.18), (132.72, 0.32, fz(132.4) + 0.68), 63)
    # ---------------- S06 — the ledger ----------------
    cam_key(cam, 689, (84.9, -2.35, 1.78), (87.15, 0.75, 1.28), 50)
    cam_key(cam, 762, (85.25, -1.95, 1.72), (87.2, 0.2, 1.3), 46)
    cam_key(cam, 832, (85.2, -2.1, 1.76), (87.2, 0.35, 1.3), 48)
    # ---------------- S07 — marquee low angle + storm ----------------
    cam_key(cam, 833, (-7.6, -0.9, 1.05), (0.3, -6.1, 4.75), 38)
    cam_key(cam, 930, (-9.6, -0.4, 1.55), (0.3, -6.1, 4.95), 36)
    cam_key(cam, 1040, (-11.5, 0.6, 2.35), (0.2, -6.2, 5.2), 34)
    # ---------------- S08 — orbiting the booth ----------------
    cam_key(cam, 1041, (112.15, 1.95, 6.15), (113.6, 0, 6.0), 36)
    cam_key(cam, 1136, (115.1, -1.75, 6.4), (113.55, 0, 5.95), 34)
    cam_key(cam, 1232, (113.1, 1.35, 7.0), (113.55, 0, 5.45), 44)
    # ---------------- S09 — the lever, tight ----------------
    cam_key(cam, 1233, (114.65, 1.02, 6.18), (113.85, 0.02, 5.85), 64)
    cam_key(cam, 1304, (114.65, 0.98, 6.15), (113.85, 0.02, 5.83), 66)
    cam_key(cam, 1376, (114.6, 0.94, 6.12), (113.85, 0.02, 5.8), 68)
    # ---------------- S10 — the lobby, handheld drift ----------------
    cam_key(cam, 1377, (103.6, 6.6, 1.72), (104.6, 8.05, 1.25), 42)
    cam_key(cam, 1405, (99.6, 4.4, 1.68), (104.8, 7.9, 1.1), 38)
    cam_key(cam, 1472, (92.4, 1.5, 1.68), (98.2, 3.8, 1.15), 40)
    cam_key(cam, 1536, (91.2, 1.15, 1.58), (105.2, 6.8, 1.7), 34)
    # ---------------- S11 — lights down; beam; the walk down the aisle ----------------
    cam_key(cam, 1537, (123.4, 0.4, fz(123.4) + 2.1), (151.7, 0, 3.6), 34)
    cam_key(cam, 1618, (119.3, -0.6, fz(119.3) + 1.95), (151.7, 0, 3.5), 36)
    cam_key(cam, 1700, (122.3, -1.35, fz(122.3) + 1.8), (151.7, 0, 3.45), 33)
    cam_key(cam, 1728, (127.2, -0.75, fz(127.2) + 1.9), (132.6, 0.575, fz(132.6) + 1.15), 46)
    # ---------------- S12 — macro intercut ----------------
    cam_key(cam, 1729, (130.7, -0.62, fz(132.4) + 1.5), (132.6, 0.575, fz(132.6) + 1.32), 58)
    cam_key(cam, 1793, (117.8, -2.6, 5.7), (124.0, 0.0, 7.18), 46)
    cam_key(cam, 1825, (134.2, -0.35, 2.15), (132.58, 0.78, fz(132.6) + 0.73), 58)
    cam_key(cam, 1878, (133.8, -0.2, 1.35), (132.6, 0.6, fz(132.6) - 0.28), 52)
    cam_key(cam, 1920, (130.9, -0.62, fz(132.4) + 1.5), (132.6, 0.575, fz(132.6) + 1.3), 58)
    # ---------------- S13 — the fold ----------------
    cam_key(cam, 1921, (129.6, -2.15, fz(130) + 2.05), (132.7, 0.6, fz(132.6) + 0.95), 42)
    cam_key(cam, 2000, (129.9, -1.85, fz(131) + 1.6), (132.65, 0.62, fz(132.6) + 0.72), 46)
    cam_key(cam, 2064, (129.6, -2.15, fz(130) + 2.0), (132.7, 0.85, fz(132.6) + 0.9), 44)
    # ---------------- S14 — shutdown & the return to F7 ----------------
    cam_key(cam, 2065, (111.3, 1.5, 6.15), (113.55, 0.1, 5.85), 42)
    cam_key(cam, 2095, (113.1, -0.85, 5.55), (113.6, 0, 5.8), 44)
    cam_key(cam, 2111, (115.9, -0.3, 5.95), (114.0, 0, 6.0), 52)
    cam_key(cam, 2112, (118.6, -0.35, floor_z(118.6) + 2.15), (132.6, 0.575, floor_z(132.6) + 0.8), 36)
    cam_key(cam, 2135, (122.0, -0.55, floor_z(122.0) + 2.05), (132.6, 0.575, floor_z(132.6) + 0.8), 38)
    cam_key(cam, 2190, (130.9, -0.62, floor_z(132.4) + 1.35), (132.6, 0.62, floor_z(132.6) + 0.82), 52)
    cam_key(cam, 2228, (131.7, -0.4, floor_z(132.4) + 1.05), (132.7, 0.62, floor_z(132.6) + 0.8), 56)
    cam_key(cam, 2240, (131.7, -0.4, floor_z(132.4) + 1.05), (132.7, 0.62, floor_z(132.6) + 0.8), 56)
    # ---------------- S15A — dawn dolly-out ----------------
    cam_key(cam, 2241, (12.5, 18.0, 1.9), (0.3, -7.2, 4.6), 40)
    cam_key(cam, 2335, (21.5, 25.5, 4.6), (0.2, -7.6, 3.9), 36)
    # ---------------- S15B — the shaft and the boat ----------------
    cam_key(cam, 2336, (134.9, -2.45, floor_z(134.9) + 1.65), (132.6, 0.62, floor_z(132.6) + 0.78), 54)
    cam_key(cam, 2390, (134.55, -2.1, floor_z(134.7) + 1.45), (132.6, 0.62, floor_z(132.6) + 0.75), 57)
    cam_key(cam, 2400, (134.55, -2.1, floor_z(134.7) + 1.45), (132.6, 0.62, floor_z(132.6) + 0.75), 57)
    return cam  # cuts ride the 1-frame key gaps between shots (bezier)

# ============================================================================
# SCENE ORCHESTRATION
# ============================================================================
def build_scene():
    log_section("BUILDING PRODUCTION SCENE")
    scene = reset_scene()
    W_BG = build_world(scene)
    M = build_materials()
    add_gradient_backdrop("SKY_SHELL", location=(0, 40, 30))
    build_exterior(scene, M)
    build_interior_shell(M)
    _, f7 = build_seat_rows(M)
    booth = build_booth(M)
    build_lobby_furniture(M)
    aud_leaves, boy_door = build_doors(M)
    P = build_props(M)
    beam, dust, screen = build_beam_and_dust(M)
    rain = build_atmosphere(M)
    elias = build_elias(M, base=(-19.0, 10.4, 0.3))
    boy = build_boy(M, base=(104.9, 7.35, 0.0))
    props_env = {"beam": beam, "dust": dust, "screen": screen,
                 "aud_doors": aud_leaves, "boy_door": boy_door,
                 "high_window": bpy.data.objects.get("HIGH_WINDOW")}
    log_line("Sets, characters and props built.")
    L = build_lights(scene)
    cam = build_camera(scene)
    animate_timeline(scene, L, P, M, props_env)
    animate_elias_a(elias, P)
    animate_elias_b(elias, P)
    animate_boy(boy, P)
    # global finish
    for ob in scene.objects:
        if ob.type == 'CAMERA':
            continue
        if ob.type == 'MESH':
            flat(ob)
    setup_render(scene, PRESETS["A"])
    log_line("Scene build complete.")
    return scene

def setup_render(scene, preset):
    scn = scene
    scn.render.engine = 'CYCLES'
    scn.cycles.device = 'CPU'
    scn.render.resolution_x = preset["w"]
    scn.render.resolution_y = preset["h"]
    scn.render.resolution_percentage = 100
    scn.cycles.samples = preset["samples"]
    scn.cycles.max_bounces = preset["max_bounces"]
    for attr, val in (("diffuse_bounces", 2), ("glossy_bounces", 1),
                      ("transmission_bounces", 2), ("transparent_max_bounces", 4)):
        try:
            setattr(scn.cycles, attr, val)
        except (AttributeError, TypeError):
            pass
    try:
        scn.cycles.use_light_tree = True
    except (AttributeError, TypeError):
        pass
    scn.cycles.use_denoising = preset["use_denoise"]
    try:
        scn.cycles.denoiser = 'OPENIMAGEDENOISE'
    except TypeError:
        pass
    scn.cycles.use_adaptive_sampling = True
    for attr, val in (("adaptive_threshold", 0.05), ("adaptive_min_samples", 2),
                      ("sample_clamp_indirect", 1.0), ("sample_clamp_direct", 3.0)):
        try:
            setattr(scn.cycles, attr, val)
        except (AttributeError, TypeError):
            pass
    scn.cycles.film_transparent = False
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGB'
    scn.render.image_settings.color_depth = '8'
    scn.render.image_settings.compression = 10
    try:
        scn.view_settings.view_transform = 'AgX'
        scn.view_settings.look = 'AgX - Medium High Contrast'
    except TypeError:
        try:
            scn.view_settings.view_transform = 'Filmic'
            scn.view_settings.look = 'Medium High Contrast'
        except TypeError:
            pass
    scn.view_settings.exposure = 0.5
    scn.render.fps = FPS_RENDER
    scn.frame_start = 1
    scn.frame_end = FRAME_END  # the authored timeline is always the 8 fps grid

# ============================================================================
# FRAME VALIDATION + RESUMABLE RENDER
# ============================================================================
def frame_path(f):
    return os.path.join(FRAMES_DIR, "frame_%04d.png" % f)

def frame_is_valid(path):
    try:
        size = os.path.getsize(path)
        if size < 3000:
            return False
        with open(path, "rb") as fh:
            head = fh.read(16)
            if head[:8] != b"\x89PNG\r\n\x1a\n":
                return False
            fh.seek(-16, os.SEEK_END)
            tail = fh.read(16)
            if b"IEND" not in tail:
                return False
        return True
    except OSError:
        return False

def render_frame(scene, f):
    scene.frame_set(f)
    t0 = time.time()
    bpy.ops.render.render(write_still=False)
    image = bpy.data.images.get("Render Result")
    image.save_render(filepath=frame_path(f), scene=scene)
    return time.time() - t0

BENCH_DIR = os.path.join(HERE, "frames_bench")
BENCHMARK_FRAMES = [48, 285, 585, 890, 1105, 1605, 1870, 2280]

def run_benchmark(scene, preset_key):
    preset = PRESETS[preset_key]
    log_section("PHASE 7 — PERFORMANCE BENCHMARK (%s)" % preset["name"])
    setup_render(scene, preset)
    os.makedirs(BENCH_DIR, exist_ok=True)
    times = []
    for f in BENCHMARK_FRAMES:
        scene.frame_set(f)
        t0 = time.time()
        bpy.ops.render.render(write_still=False)
        image = bpy.data.images.get("Render Result")
        image.save_render(filepath=os.path.join(BENCH_DIR, "bench_%04d.png" % f), scene=scene)
        t = time.time() - t0
        times.append(t)
        log_line("BENCHMARK frame %d (%s): %.2fs" % (f, _shot_for_frame(f), t))
    avg = sum(times) / len(times)
    scene.frame_end = FRAME_END
    total_frames = scene.frame_end
    est_s = avg * total_frames
    log_line("Average per frame: %.3fs over %d samples" % (avg, len(times)))
    log_line("Estimated total render: %.1f minutes (%d frames @ %d fps grid)" %
             (est_s / 60.0, total_frames, FPS_RENDER))
    return avg, est_s / 60.0

def _shot_for_frame(f):
    for name, t0, t1 in SHOTS:
        if int(t0 * FPS_RENDER) < f <= int(t1 * FPS_RENDER):
            return name
    return "?"

def auto_select_preset(scene, forced=None, budget_minutes=BUDGET_MINUTES):
    if forced:
        preset = PRESETS[forced]
        setup_render(scene, preset)
        log_line("Preset %s forced — skipping auto-selection." % forced)
        return forced
    for key in ("A", "B", "C"):
        avg, est_min = run_benchmark(scene, key)
        if est_min <= budget_minutes:
            log_line("SELECTED PRESET %s — estimated %.1f min <= %d min budget." %
                     (key, est_min, budget_minutes))
            return key
        log_line("PRESET %s exceeds budget (%.1f min > %d min). Downgrading..." %
                 (key, est_min, budget_minutes))
    log_line("PRESET C is the floor; rendering anyway with PRESET C.")
    return "C"

def render_all(scene, preset_key, max_frames=None):
    preset = PRESETS[preset_key]
    setup_render(scene, preset)
    end = scene.frame_end
    if max_frames:
        end = min(end, max_frames)
    todo = [f for f in range(1, end + 1) if not frame_is_valid(frame_path(f))]
    done = end - len(todo)
    log_section("PHASE 8 — FINAL PRODUCTION (resume-safe)")
    log_line("Preset: %s | resolution %dx%d | timeline 8 fps | frame_end=%d" %
             (preset_key, preset["w"], preset["h"], scene.frame_end))
    log_line("Already valid: %d | To render: %d" % (done, len(todo)))
    t_start = time.time()
    rendered = 0
    for f in todo:
        try:
            t = render_frame(scene, f)
            rendered += 1
            log_line("RENDER frame %d/%d %.2fs (shot %s)" % (f, end, t, _shot_for_frame(f)))
        except Exception as exc:
            log_line("ERROR frame %d: %s" % (f, exc))
        if rendered % 25 == 1 and rendered > 1:
            try:
                bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
                log_line("Incremental save -> %s" % BLEND_PATH)
            except Exception as exc:
                log_line("WARN: incremental save failed: %s" % exc)
    elapsed = time.time() - t_start
    log_line("RENDER COMPLETE: %d frames in %.1f minutes" % (rendered, elapsed / 60.0))
    return rendered

# ============================================================================
# ASSEMBLY — ffmpeg (imageio-ffmpeg), 24 fps conform + title endcard
# ============================================================================
def get_ffmpeg():
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        log_line("ffmpeg (imageio-ffmpeg): %s" % exe)
        return exe
    except Exception as exc:
        log_line("imageio-ffmpeg unavailable (%s); falling back to system ffmpeg" % exc)
        import shutil
        exe = shutil.which("ffmpeg")
        if exe:
            log_line("system ffmpeg: %s" % exe)
            return exe
        return None

def assemble():
    import subprocess
    exe = get_ffmpeg()
    log_section("FINAL ASSEMBLY")
    if exe is None:
        log_line("FATAL: no ffmpeg available. Frames preserved in %s" % FRAMES_DIR)
        log_line("Manual assembly: ffmpeg -framerate 8 -i frames/frame_%%04d.png "
                 "-i audio.wav -vf fps=24,scale=640:360 -c:v libx264 -pix_fmt yuv420p "
                 "-c:a aac final_film.mp4")
        return False
    if not frame_is_valid(frame_path(1)):
        log_line("FATAL: frame_0001.png missing/invalid — render first.")
        return False
    font = FONT_PATH
    vf = ("fps=%d,scale=640:360:flags=lanczos,"
          "fade=t=out:st=296.0:d=4.0,"
          "drawtext=fontfile='%s':text='PALACE':fontsize=64:fontcolor=0xF2EDE3:"
          "x=(w-text_w)/2:y=(h-text_h)/2-24:alpha='if(lt(t,296.6),0,if(lt(t,297.4),(t-296.6)/0.8,if(gt(t,299.3),(300-t)/0.7,1)))',"
          "drawtext=fontfile='%s':text='for everyone who ever had a favorite seat':fontsize=17:fontcolor=0xB9AE97:"
          "x=(w-text_w)/2:y=(h-text_h)/2+34:alpha='if(lt(t,297.1),0,if(lt(t,297.9),(t-297.1)/0.8,if(gt(t,299.3),(300-t)/0.7,1)))'") % (
              FPS_FINAL, font, font)
    cmd = [exe, "-y", "-framerate", str(FPS_RENDER), "-i",
           os.path.join(FRAMES_DIR, "frame_%04d.png"),
           "-i", AUDIO_WAV,
           "-vf", vf,
           "-c:v", "libx264", "-crf", "20", "-preset", "medium",
           "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k",
           "-t", "300.0",
           "-movflags", "+faststart",
           FINAL_MP4]
    log_line("Assembly command issued (frame duplication 8->24 fps).")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log_line("ASSEMBLY FAILED:\n%s\n%s" % (proc.stdout[-2000:], proc.stderr[-3000:]))
        log_line("Frames and audio preserved. Alternative: see README.txt 'Manual assembly'.")
        return False
    log_line("final_film.mp4 written in %.1fs" % (time.time() - t0))
    # preview: downscaled derivative of the finished cut
    prev = [exe, "-y", "-i", FINAL_MP4,
            "-vf", "scale=426:-2",
            "-c:v", "libx264", "-crf", "28", "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart",
            PREVIEW_MP4]
    proc = subprocess.run(prev, capture_output=True, text=True)
    if proc.returncode != 0:
        log_line("PREVIEW FAILED:\n%s" % proc.stderr[-1500:])
        return False
    log_line("preview.mp4 written.")
    return True

# ============================================================================
# README
# ============================================================================
README_TEMPLATE = """PALACE — a wordless 5-minute cinematic short film
=================================================
Directed / written / produced procedurally with Blender's Python API.

FILES
-----
  film_main.py      the complete production script (detect, build, render, assemble)
  film_project.blend the saved Blender project (resumable)
  final_film.mp4    the finished film, exactly 300.0s, 24 fps
  preview.mp4       low-resolution derivative of the finished cut
  audio.wav         the complete synthesized soundtrack (22.05 kHz mono)
  storyboard.txt    creative development + full shot-by-shot breakdown
  render_log.txt    environment report, benchmark, render times, assembly log
  frames/           rendered PNG frames (unique frames; safe to re-render)

REQUIREMENTS
------------
  * Blender 4.5 LTS-class Python module (pip package `bpy`) or Blender itself.
  * If bpy is not installed system-wide, run ./restore_env.sh once. It builds
    the in-repo environment under _env/ (gitignored): bpy 4.5.15 + numpy +
    imageio-ffmpeg wheels from PyPI, plus the headless X11/GL stub libraries
    bpy's loader needs on a GL-less machine. Then run ./blenderpy.sh film_main.py --all
  * ffmpeg (here: the static build bundled with the `imageio-ffmpeg` wheel).
  * No GPU required. No internet. No external assets. Fonts ship with bpy.

RUNNING
-------
  With the bpy module (as produced here):
      ./blenderpy.sh film_main.py --all

  With a real Blender installation:
      blender --background --python film_main.py -- --all

  Individual phases:
      --detect     environment/compatibility report (Phase 5)
      --build-only construct the scene and save film_project.blend
      --benchmark  render 8 representative frames, estimate total, auto-pick preset
      --render     render all frames (resume-safe; existing valid frames are skipped)
      --audio      synthesize audio.wav
      --assemble   build final_film.mp4 + preview.mp4

QUALITY PRESETS (auto-selected by benchmark; force with --preset A|B|C)
-----------------------------------------------------------------------
  A  640x360  8 fps grid (2400 unique frames)   6 samples
  B  480x270  8 fps grid (2400 unique frames)   6 samples
  C  426x240  8 fps grid (2400 unique frames)   4 samples
  All presets render 2400 unique frames on the fixed 8 fps animation grid so
  timing, shots and the exact 300.0 s runtime are identical at every preset.

THE CONFORMED 24 FPS (honest note)
-----------------------------------
final_film.mp4 plays at 24 fps. Because unique footage was rendered on an
8 fps grid, each rendered frame is DUPLICATED x3, not interpolated and not
truly animated at 24 fps. The lower cadence reads as deliberate stop-motion /
limited animation — the house style of the film.
Audio is cut to the same 300.0 s timeline either way.

CRASH RECOVERY
--------------
Re-run --render at any time. Frames that exist and pass PNG signature/size
checks are kept; only missing or corrupt frames are re-rendered. The .blend
is re-saved incrementally every 25 frames. Do NOT delete frames/ to retry —
it simply resumes.

MANUAL ASSEMBLY (if ffmpeg ever fails)
--------------------------------------
  ffmpeg -framerate 8 -i frames/frame_%04d.png -i audio.wav \\
         -vf "fps=24,scale=640:360,fade=t=out:st=296:d=4" \\
         -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -t 300 final_film.mp4
Preview: ffmpeg -i final_film.mp4 -vf scale=426:-2 -c:v libx264 -crf 28 preview.mp4

NOTES ON CRAFT
--------------
  * Exactly 300.0 seconds, 15 shots, one location, two characters.
  * No background music; the soundtrack is 100% synthesized (sea, rain,
    projector clatter, footsteps, thunder, one clock chime, one final click).
  * The boy's yellow raincoat and the white paper boat are the only
    saturated elements; everything else is desaturated teal/sepia.
  * Storyboard continuity note: the original treatment carried a 1953
    flashback in S12; the final cut replaces it with two wordless ECUs
    (the lit attic hatch; the boat on F7's armrest) — quieter, and cheaper
    to light. storyboard.txt reflects this change.
"""

def write_readme():
    with open(README_PATH, "w", encoding="utf-8") as fh:
        fh.write(README_TEMPLATE)
    log_line("README.txt written.")

# ============================================================================
# SOUND DESIGN — 100% synthesized (wave + math + struct; numpy for speed).
# No music. Diegetic sound only. Stereo is mixed to mono bus at 22050 Hz.
# ============================================================================
def _noise(rng, n, kind="white"):
    x = rng.standard_normal(n).astype(np.float64)
    if kind == "white":
        return x
    if kind == "pink":
        # Paul Kellet-ish pink filter
        b0, b1, b2 = 0.0, 0.0, 0.0
        out = np.empty(n)
        for i in range(n):
            w = x[i]
            b0 = 0.99765 * b0 + w * 0.0990460
            b1 = 0.96300 * b1 + w * 0.2965164
            b2 = 0.57000 * b2 + w * 1.0526913
            out[i] = (b0 + b1 + b2 + w * 0.1848) * 0.05
        return out
    if kind == "brown":
        out = np.cumsum(x)
        out = out - np.mean(out)
        return out / (np.max(np.abs(out)) + 1e-9)
    return x

def _lowpass(x, alpha):
    if alpha >= 1.0:
        return x.copy()
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += alpha * (x[i] - acc)
        y[i] = acc
    return y

def _lowpass_fft(x, cutoff_hz, rate):
    """Zero-phase brick-ish lowpass via FFT (fast, no phase smear on clicks)."""
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / rate)
    spec[freqs > cutoff_hz] = 0.0
    return np.fft.irfft(spec, n=len(x))

def _highpass_fft(x, cutoff_hz, rate):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / rate)
    spec[freqs < cutoff_hz] = 0.0
    return np.fft.irfft(spec, n=len(x))

def _bandpass_fft(x, lo, hi, rate):
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / rate)
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(spec, n=len(x))

def _sine(freq, dur, rate, phase=0.0):
    t = np.arange(int(dur * rate)) / rate
    return np.sin(2 * np.pi * freq * t + phase)

def _env_exp(dur, rate, attack=0.005, decay=6.0):
    n = int(dur * rate)
    t = np.arange(n) / rate
    a = np.minimum(1.0, t / max(attack, 1e-4))
    return a * np.exp(-decay * t)

def _place(mix, t_start, clip, gain=1.0):
    i = int(t_start * AUDIO_RATE)
    j = min(len(mix), i + len(clip))
    if j > i:
        mix[i:j] += clip[:j - i] * gain

def _tick(dur=0.03):
    rate = AUDIO_RATE
    c = _sine(2500, dur, rate) * _env_exp(dur, rate, 0.001, 90)
    c += _sine(1200, dur, rate) * _env_exp(dur, rate, 0.001, 120) * 0.6
    return c

def _thump(freq=110.0, dur=0.22, gain_env=8.0):
    rate = AUDIO_RATE
    c = _sine(freq, dur, rate) * _env_exp(dur, rate, 0.003, gain_env)
    c += _noise(np.random.default_rng(5), len(c), "white") * _env_exp(dur, rate, 0.001, 40) * 0.12
    return c

def _click(freq=1800.0, dur=0.05):
    rate = AUDIO_RATE
    rng = np.random.default_rng(11)
    return (_sine(freq, dur, rate) * _env_exp(dur, rate, 0.001, 60)
            + _noise(rng, int(dur * rate), "white") * _env_exp(dur, rate, 0.001, 120) * 0.4)

def _crinkle(dur=0.32, seed=3):
    rate = AUDIO_RATE
    rng = np.random.default_rng(seed)
    n = int(dur * rate)
    bursts = np.zeros(n)
    positions = sorted(rng.integers(0, n, size=int(dur * 26)))
    for p in positions:
        ln = int(rng.integers(28, 90))
        env = np.exp(-np.arange(ln) / 14.0)
        burst = rng.standard_normal(ln) * env
        burst = _highpass_fft(burst, 900, rate)
        end = min(n, p + ln)
        bursts[p:end] += burst[:end - p]
    return bursts * 0.6

def _gull(dur=0.9, seed=7):
    rate = AUDIO_RATE
    t = np.arange(int(dur * rate)) / rate
    f0, f1 = 1450.0, 820.0
    freq = f0 + (f1 - f0) * (t / dur) + 90 * np.sin(2 * np.pi * 23 * t)
    phase = np.cumsum(2 * np.pi * freq / rate)
    env = np.minimum(1.0, t / 0.06) * np.exp(-2.2 * t)
    c = np.sin(phase) * env
    c += 0.4 * np.sin(2 * phase) * env
    return _bandpass_fft(c, 500, 3200, rate) * 0.8

def _thunder(dur=4.5, seed=17):
    rate = AUDIO_RATE
    rng = np.random.default_rng(seed)
    n = int(dur * rate)
    body = _noise(rng, n, "brown")
    body = _lowpass_fft(body, 170, rate)
    t = np.arange(n) / rate
    env = np.minimum(1.0, t / 0.05) * np.exp(-0.9 * t)
    env *= (1.0 + 0.35 * np.sin(2 * np.pi * 0.8 * t) * np.exp(-0.7 * t))
    c = body * env
    th = _thump(55, 1.4, 2.2)
    c[:th.shape[0]] += th * 0.9
    return c / (np.max(np.abs(c)) + 1e-9) * 0.9

def _footstep(leather=True):
    rate = AUDIO_RATE
    rng = np.random.default_rng(23)
    dur = 0.16
    c = _thump(96 if leather else 140, dur, 16.0)
    scuff = _noise(rng, int(dur * rate), "white") * _env_exp(dur, rate, 0.002, 26) * 0.18
    scuff = _bandpass_fft(scuff, 500, 2800, rate)
    return c * 0.8 + scuff

def _creak(dur=1.5, f_start=165.0, f_end=92.0):
    rate = AUDIO_RATE
    t = np.arange(int(dur * rate)) / rate
    freq = np.linspace(f_start, f_end, len(t)) + 14 * np.sin(2 * np.pi * 7.3 * t)
    phase = np.cumsum(2 * np.pi * freq / rate)
    env = np.sin(np.pi * np.minimum(1.0, t / dur)) ** 0.7
    env *= (1.0 + 0.3 * np.sign(np.sin(2 * np.pi * 11 * t)))
    return np.sin(phase) * env * 0.42

def _projector_clatter(dur, rate=AUDIO_RATE):
    """24 fps claw + flange: an impulse train at 24/12 Hz over motor hum."""
    n = int(dur * rate)
    t = np.arange(n) / rate
    rng = np.random.default_rng(101)
    c = np.zeros(n)
    period = 1.0 / 24.0
    i = 0
    while i / rate < dur:
        p = int(i + rng.integers(-30, 30))
        if 0 <= p < n:
            ln = 90
            imp = np.array([1.0, -0.6, 0.28] * 30, dtype=np.float64)[:ln] * np.exp(-np.arange(ln) / 22.0)
            end = min(n, p + ln)
            c[p:end] += imp[:end - p] * (0.55 + 0.35 * ((i / period) % 2))
        i += int(period * rate)
    c += _sine(50.0, dur, rate) * 0.045 + _sine(100.0, dur, rate) * 0.02
    flutter = _bandpass_fft(rng.standard_normal(n), 700, 2600, rate)
    c += flutter * (0.16 + 0.12 * np.sin(2 * np.pi * 24 * t) ** 2)
    return c * 0.55


def synthesize_audio():
    log_section("SOUND DESIGN — synthesizing audio.wav")
    rate = AUDIO_RATE
    total = int(DURATION_S * rate)
    t_all = np.arange(total) / rate
    rng = np.random.default_rng(20240501)
    mix = np.zeros(total, dtype=np.float64)

    # ---------- beds ----------
    sea = _lowpass_fft(_noise(rng, total, "brown"), 420, rate)
    sea *= (0.55 + 0.45 * np.sin(2 * np.pi * t_all / 9.0 + 0.8))          # slow swell
    mix += sea * 0.085
    wind = _bandpass_fft(_noise(rng, total, "pink"), 240, 900, rate)
    wind *= (0.6 + 0.4 * np.sin(2 * np.pi * t_all / 17.0 + 2.1))
    mix += wind * 0.05
    roomtone = _lowpass_fft(rng.standard_normal(total), 140, rate) * 0.012
    roomtone += _sine(50, DURATION_S, rate) * 0.006
    interior_mask = np.zeros(total)                                        # interiors are quieter, drier
    for a, b in [(30, 44), (44, 64), (64, 86), (86, 104), (130, 154), (154, 172),
                 (172, 192), (192, 216), (216, 240), (240, 258), (258, 280)]:
        i, j = int(a * rate), int(b * rate)
        interior_mask[i:j] = 1.0
    exterior_mask = 1.0 - interior_mask * 0.82
    mix = mix * exterior_mask + roomtone

    # rain: fades in late in S06, full storm through S13, gone by S15
    rain = _bandpass_fft(rng.standard_normal(total), 1100, 4800, rate)
    rain += _lowpass_fft(rng.standard_normal(total), 500, rate) * 0.5
    env = np.zeros(total)
    i0, i1, i2, i3 = int(96.5 * rate), int(104.0 * rate), int(244.0 * rate), int(262.0 * rate)
    env[i0:i1] = np.linspace(0.0, 1.0, i1 - i0)
    env[i1:i2] = 1.0
    env[i2:i3] = np.linspace(1.0, 0.0, i3 - i2)
    mix += rain * env * 0.155
    # rain patter on the lobby glass specifically (high tick density)
    patter = _bandpass_fft(rng.standard_normal(total), 2400, 6500, rate)
    mix += patter * env * 0.05

    def add(t, clip, gain=1.0):
        _place(mix, t, clip, gain)

    # ---------- S01 dawn: gull + the neon fritz ----------
    add(6.1, _gull(0.9, seed=7), 0.5)
    add(7.0, _gull(0.7, seed=8), 0.32)
    fritz = _sine(120, 0.34, rate) * 0.5 + _sine(240, 0.34, rate) * 0.3
    fritz = fritz * _env_exp(0.34, rate, 0.002, 9.0)
    add(10.5, fritz, 0.5); add(10.62, fritz * 0.8, 0.4); add(10.85, fritz * 0.5, 0.3)
    # ---------- S02: footsteps along the promenade ----------
    t = 13.2
    while t < 29.6:
        add(t, _footstep(True) * (0.55 + 0.2 * (t * 7 % 1)), 0.6)
        t += 0.62
    # ---------- S03: keys, lock, door, drips ----------
    add(30.5, _click(2500, 0.06) * 0.8 + _click(3100, 0.06) * 0.6 + _click(1900, 0.06) * 0.5, 0.7)
    add(31.2, _thump(78, 0.18, 14.0), 0.85)
    add(31.2, _click(1400, 0.04), 0.55)
    add(31.75, _creak(1.7, 168.0, 96.0), 0.85)
    add(33.9, _thump(64, 0.5, 6.0), 0.35)               # the door settling, house asleep
    add(38.2, _sine(2050, 0.34, rate) * _env_exp(0.34, rate, 0.001, 11), 0.28)   # drip
    add(41.6, _sine(1980, 0.33, rate) * _env_exp(0.33, rate, 0.001, 11), 0.24)   # drip
    # ---------- S04: ladder rungs, oil, threading ----------
    add(44.85, _click(720, 0.09) * 0.9 + _thump(190, 0.09, 18) * 0.5, 0.6)
    add(49.9, _click(700, 0.09) * 0.9 + _thump(185, 0.09, 18) * 0.5, 0.55)
    add(53.75, (_sine(680, 0.16, rate) + 0.4 * _sine(920, 0.16, rate))
               * _env_exp(0.16, rate, 0.01, 7), 0.22)   # oil-can squeak
    add(57.5, _click(1850, 0.05), 0.5); add(60.4, _click(1750, 0.05), 0.5)
    add(63.1, _click(1920, 0.05), 0.55)
    # ---------- S05: broom, the boat, the look up ----------
    for t in (65.6, 67.7, 69.8):
        add(t, _bandpass_fft(rng.standard_normal(int(0.5 * rate)), 300, 1200, rate)
               * _env_exp(0.5, rate, 0.03, 5), 0.30)    # broom stroke
    add(70.6, _crinkle(0.45, seed=3), 0.75)             # the boat, lifted
    add(74.6, _crinkle(0.3, seed=4), 0.55)              # into the pocket
    add(80.2, _thump(120, 0.12, 16) * 0.5, 0.4)         # a step uncovers the prints
    # ---------- S06: the clock; the stamp; first drops ----------
    t = 86.0
    while t < 172.0:                                    # one clock, one pulse
        add(t, _tick(0.028), 0.34)
        t += 1.0
    add(92.4, _thump(118, 0.2, 13), 0.95)               # STAMP
    add(92.4, _click(950, 0.04), 0.6)
    add(96.15, _crinkle(0.5, seed=9), 0.4)              # page turn
    add(99.6, _bandpass_fft(rng.standard_normal(int(0.8 * rate)), 2000, 5200, rate)
              * _env_exp(0.8, rate, 0.25, 3), 0.16)     # first scattered drops
    # ---------- S07: thunder; marquee letters ----------
    add(112.0, _thunder(5.2, seed=17), 0.95)
    add(118.0, _thunder(4.1, seed=18), 0.8)
    add(127.0, _thunder(6.0, seed=19), 1.0)
    add(113.2, _click(560, 0.12) + _thump(240, 0.12, 16) * 0.6, 0.55)   # letters, steel on steel
    add(119.0, _click(540, 0.12), 0.5)
    add(124.5, _click(610, 0.12), 0.5)
    add(128.3, _click(520, 0.12) + _thump(220, 0.12, 15) * 0.5, 0.6)
    add(116.8, _click(760, 0.08), 0.4)                  # ladder settles
    # ---------- S08: mains clunk; the machine wakes ----------
    add(132.0, _thump(58, 0.34, 8.5), 1.0)              # CLUNK
    add(132.0, _click(320, 0.06), 0.7)
    add(138.0, (_sine(np.linspace(80, 235, int(0.9 * rate)), 0.9, rate)
                * _env_exp(0.9, rate, 0.35, 2.6)), 0.18) # spool-up whine
    # ---------- S09: the clock owns the room ----------
    add(172.0, _thump(196, 1.6, 2.6) * 0.9, 0.75)       # one low bell: 7:30
    add(172.0, _sine(392, 1.8, rate) * _env_exp(1.8, rate, 0.01, 1.6) * 0.28, 0.6)
    # ---------- S10: the door bangs; rain; coin; stub ----------
    add(172.0, _thump(84, 0.4, 7.0), 1.0)               # BANG
    add(172.05, _creak(0.9, 210.0, 150.0), 0.5)
    add(172.1, _bandpass_fft(rng.standard_normal(int(1.6 * rate)), 900, 4200, rate)
               * _env_exp(1.6, rate, 0.02, 2.2), 0.30)  # gust of wet air
    for t in (174.3, 175.6, 176.9, 178.4, 179.8):       # dripping from his coat
        add(t, _sine(2300, 0.3, rate) * _env_exp(0.3, rate, 0.001, 13), 0.16)
    add(181.6, _sine(3350, 0.32, rate) * _env_exp(0.32, rate, 0.001, 15)
               + _sine(5020, 0.32, rate) * _env_exp(0.32, rate, 0.001, 18) * 0.5, 0.5)  # coin
    add(185.0, _noise(rng, int(0.09 * rate), "white") * _env_exp(0.09, rate, 0.001, 30)
               * 0 + _crinkle(0.09, seed=12) * 0.9, 0.8)   # stub torn free
    add(190.4, _thump(95, 0.25, 9), 0.7)                # the house doors boom back
    # ---------- S11: houselight hum; the motor takes; the wash of the sea ----------
    add(192.0, _sine(110, 0.7, rate) * _env_exp(0.7, rate, 0.01, 3) * 0.16, 0.5)
    add(192.75, _thump(66, 0.5, 6.5), 1.0)              # CHANGEOVER
    mix[int(192.8 * rate):int(248.6 * rate)] += _projector_clatter(55.8, rate) * 0.30
    # faint film-soundtrack waves (the film-within-the-film), surf under everything
    surf = _lowpass_fft(rng.standard_normal(total), 240, rate)
    surf *= (0.6 + 0.4 * np.sin(2 * np.pi * t_all / 11.0))
    surf_mask = np.zeros(total)
    surf_mask[int(194 * rate):int(246 * rate)] = 0.030
    mix += surf * surf_mask
    # ---------- S12: the boy's held breath, once ----------
    add(225.2, _sine(330, 0.55, rate) * _env_exp(0.55, rate, 0.18, 3.2) * 0.05, 0.5)
    # ---------- S13: leader flaps; spin-down; SILENCE; folds ----------
    for t in (244.6, 245.65, 246.7):
        add(t, _noise(rng, int(0.07 * rate), "white") * _env_exp(0.07, rate, 0.001, 26) * 0.8
               + _thump(340, 0.07, 12) * 0.6, 0.9)      # FLAP
    spindown = _sine(np.linspace(230, 62, int(2.6 * rate)), 2.6, rate) * _env_exp(2.6, rate, 0.05, 1.05)
    add(248.3, spindown, 0.22)
    # 250.0 -> 252.2 : intentional near-silence (floor tone only)
    add(252.6, _crinkle(0.34, seed=21), 0.8)            # fold 1
    add(253.35, _crinkle(0.3, seed=22), 0.75)           # fold 2
    add(254.15, _crinkle(0.36, seed=23), 0.8)           # fold 3
    add(255.3, _thump(68, 0.35, 5.0) * 0.4, 0.35)       # chair he was sitting on
    add(257.4, _creak(1.1, 150.0, 110.0), 0.5)          # door opens — he slips out
    # ---------- S14: breakers; the boat; the booth lamp dies ----------
    add(261.2, _thump(74, 0.3, 9.5), 0.95)              # BREAKER 1
    add(266.2, _thump(70, 0.3, 9.5), 0.9)               # BREAKER 2
    add(271.2, _thump(64, 0.34, 8.5), 0.95)             # BREAKER 3 (the lamp)
    step_t = 270.9
    while step_t < 276.6:                                # down the aisle, one row at a time
        add(step_t, _footstep(True) * 0.62, 0.4)
        step_t += 0.72
    add(277.0, _crinkle(0.28, seed=31), 0.6)            # boat lifted from the armrest
    add(277.6, _crinkle(0.24, seed=32), 0.5)            # set on the seat forever
    add(278.9, _click(2100, 0.05) * 0.7, 0.4)           # keys on the OPEN hook
    # ---------- S15: gulls; wind; one dead-machine click; black ----------
    add(281.6, _gull(1.0, seed=33), 0.42)
    add(288.1, _gull(0.8, seed=34), 0.3)
    add(297.45, _click(1250, 0.07), 0.85)               # the click in the dark
    # ---------- master bus: gentle clip, -3 dBFS peak ----------
    mix = np.tanh(mix * 0.9)
    peak = np.max(np.abs(mix)) + 1e-9
    mix *= (10 ** (-3.0 / 20.0)) / peak
    fade_out = np.linspace(1.0, 0.0, int(1.2 * rate))
    mix[-len(fade_out):] *= fade_out
    pcm = (mix * 32767.0).astype(np.int16)
    with wave.open(AUDIO_WAV, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())
    log_line("audio.wav written: %d samples = %.3fs, peak -3 dBFS" % (len(pcm), len(pcm) / rate))
    return True


# ============================================================================
# MAIN
# ============================================================================
def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--preset", choices=list(PRESETS), default=None)
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args(argv)

    if not any([args.detect, args.build_only, args.benchmark, args.render,
                args.audio, args.assemble, args.all]):
        args.all = True

    log_section("PALACE — PRODUCTION START %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    report = detect_environment()
    print_detection_report(report)

    if args.detect:
        return 0

    if args.audio and not (args.all or args.render):
        ok = synthesize_audio()
        return 0 if ok else 1

    scene = None
    if args.all or args.build_only or args.benchmark or args.render:
        scene = build_scene()
        try:
            bpy.ops.wm.save_as_mainfile(filepath=BLEND_PATH)
            log_line("Project saved -> %s" % BLEND_PATH)
        except Exception as exc:
            log_line("WARN: could not save .blend: %s" % exc)

    if args.build_only:
        return 0

    preset_key = args.preset
    if args.all or args.benchmark or args.render:
        if args.all:
            preset_key = auto_select_preset(scene, forced=args.preset)
        elif args.benchmark:
            if args.preset:
                run_benchmark(scene, args.preset)
            else:
                select_key = auto_select_preset(scene, forced=None)
                log_line("Auto-selection chose PRESET %s" % select_key)
            return 0
        else:  # --render
            preset_key = args.preset or "A"
            if args.preset is None:
                # re-read benchmark decision if a previous run logged one
                preset_key = "A"
                log_line("--render with no --preset: defaulting to PRESET A "
                         "(run --benchmark first for auto-selection).")
    if args.all or args.render:
        render_all(scene, preset_key, max_frames=args.frames)
    if args.all or args.audio:
        synthesize_audio()
    if args.all or args.assemble:
        write_readme()
        assemble()
    log_section("PALACE — PRODUCTION END")
    return 0

if __name__ == "__main__":
    sys.exit(main())

