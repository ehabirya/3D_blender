#!/usr/bin/env python3
"""
deform_avatar.py - Blender avatar generation script

Pipeline:
1) Load/prepare mesh and measurements
2) Build multi-photo projection material + FaceMask
3) Bake diffuse in neutral pose (OBJECT mode, safe active image)
4) Save, reload, and pack baked texture
5) Optional pose AFTER bake
6) Robust GLB export with verification

Notes:
- Honors env OUTPUT_GLTF first, then --out, then /tmp/avatar.glb
- Logs helpful details for serverless runs
"""

import bpy
import os
import sys
import argparse
import math
import json
import time
import numpy as np
from pathlib import Path

print("=" * 80)
print("[DEFORM] Blender Avatar Generation Script Starting...")
print(f"[DEFORM] Blender version: {bpy.app.version_string}")
print(f"[DEFORM] Python version: {sys.version}")
print(f"[DEFORM] CWD: {os.getcwd()}")
print("=" * 80)

# -------------------------- CLI --------------------------
parser = argparse.ArgumentParser(description="Generate 3D avatar with Blender")
parser.add_argument("--preset", type=str, default="neutral")
parser.add_argument("--height", type=float, required=True)
parser.add_argument("--chest", type=float)
parser.add_argument("--waist", type=float)
parser.add_argument("--hips", type=float)
parser.add_argument("--shoulder", type=float)
parser.add_argument("--inseam", type=float)
parser.add_argument("--arm", type=float)
parser.add_argument("--frontTex", type=str)
parser.add_argument("--sideTex", type=str)
parser.add_argument("--backTex", type=str)
parser.add_argument("--frontTexList", type=str, default="")
parser.add_argument("--sideTexList", type=str, default="")
parser.add_argument("--backTexList", type=str, default="")
parser.add_argument("--texRes", type=int, default=2048)
parser.add_argument("--highDetail", action="store_true")
parser.add_argument("--poseJson", type=str, default="")
parser.add_argument("--out", type=str, default="/tmp/avatar.glb")
parser.add_argument("--make_bases", action="store_true")

if "--" in sys.argv:
    args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:])
else:
    args, _ = parser.parse_known_args()

print("[DEFORM] Args:")
for k, v in vars(args).items():
    if k not in {"frontTex", "sideTex", "backTex"}:
        print(f"  - {k}: {v}")

# -------------------- Output path --------------------
OUTPUT_GLTF = os.environ.get("OUTPUT_GLTF") or args.out or "/tmp/avatar.glb"
out_dir = os.path.dirname(OUTPUT_GLTF) or "/tmp"
Path(out_dir).mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("[DEFORM] Output Path Configuration")
print(f"  ENV OUTPUT_GLTF: {os.environ.get('OUTPUT_GLTF','<unset>')}")
print(f"  CLI --out      : {args.out}")
print(f"  Final path     : {OUTPUT_GLTF}")
print("=" * 80)

# -------------------- Make bases (optional) --------------------
if args.make_bases:
    print("[DEFORM] Generating base .blend files...")
    base_dir = "/app/assets"
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    for cat in ["male", "female", "neutral", "child", "baby"]:
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
        bpy.context.active_object.name = f"Base_{cat}"
        bpy.ops.object.shade_smooth()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(base_dir, f"base_{cat}.blend"))
        print(f"  saved {cat}")
    print("[DEFORM] Done.")
    sys.exit(0)

# -------------------------- Helpers --------------------------
def to_object_mode():
    try:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

def _main_mesh():
    ms = [o for o in bpy.data.objects if o.type == "MESH"]
    if not ms:
        print("[DEFORM] No mesh found; creating UV sphere.")
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
        return bpy.context.active_object
    return ms[0]

def _ensure_uv(obj):
    to_object_mode()
    if not obj.data.uv_layers:
        print("[DEFORM] Creating UVs…")
        try:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.uv.smart_project(angle_limit=66)
        finally:
            to_object_mode()

def _load_img_node(nodes, links, path):
    if not path or not os.path.exists(path):
        if path: print(f"[DEFORM] Missing image: {path}")
        return None
    img = bpy.data.images.load(path)
    tex = nodes.new("ShaderNodeTexImage"); tex.image = img; tex.extension = 'CLIP'
    tcoord = nodes.get("TexCoord") or nodes.new("ShaderNodeTexCoord"); tcoord.name = "TexCoord"
    links.new(tcoord.outputs["Camera"], tex.inputs["Vector"])
    return tex

def _split_list(arg): 
    return [p for p in (arg or "").split(";") if p and os.path.exists(p)]

def _load_many(nodes, links, paths): 
    return [n for p in paths if (n := _load_img_node(nodes, links, p))]

def _merge_images(nodes, links, tex_nodes, weights=None):
    if not tex_nodes: return None
    if len(tex_nodes) == 1: return tex_nodes[0].outputs["Color"]
    weights = weights or [1.0]*len(tex_nodes)
    s = sum(weights) or 1.0
    weights = [w/s for w in weights]
    acc_color = tex_nodes[0].outputs["Color"]; acc_w = weights[0]
    for tex, w in zip(tex_nodes[1:], weights[1:]):
        mix = nodes.new("ShaderNodeMixRGB")
        denom = nodes.new("ShaderNodeMath"); denom.operation='ADD'
        denom.inputs[0].default_value = acc_w; denom.inputs[1].default_value = w
        div = nodes.new("ShaderNodeMath"); div.operation='DIVIDE'
        links.new(denom.outputs["Value"], div.inputs[1]); div.inputs[0].default_value = w
        links.new(div.outputs["Value"], mix.inputs["Fac"])
        links.new(acc_color, mix.inputs[1]); links.new(tex.outputs["Color"], mix.inputs[2])
        acc_color = mix.outputs["Color"]; acc_w += w
    return acc_color

def _make_facemask_if_missing(obj, name="FaceMask"):
    if obj.data.attributes.get(name): return name
    print("[DEFORM] Building FaceMask attribute…")
    attr = obj.data.attributes.new(name=name, type='FLOAT', domain='POINT')
    verts = obj.data.vertices
    if not verts: return name
    zs = np.array([v.co.z for v in verts], dtype=np.float32)
    z_min, z_max = float(zs.min()), float(zs.max())
    z_low = z_min + 0.65*(z_max - z_min); z_high = z_min + 0.95*(z_max - z_min)
    obj.data.calc_normals()
    vals = np.zeros(len(verts), dtype=np.float32)
    for i, v in enumerate(verts):
        ny = v.normal.y; z = v.co.z
        head_band = 0.0
        if z >= z_low: head_band = min(1.0, max(0.0, (z - z_low)/max(1e-6, (z_high - z_low))))
        frontness = 1.0 if ny < -0.15 else 0.0
        vals[i] = float(head_band*frontness)
    if vals.max() > 0: vals = (vals/vals.max())**0.7
    for i, d in enumerate(attr.data): d.value = float(vals[i])
    return name

def _set_simple_shape(obj):
    scale = max(0.2, float(args.height)/1.75)
    obj.scale = (scale, scale, scale)
    print(f"[DEFORM] Scale set ~ {scale:.3f}")

def ensure_basic_armature(body_obj):
    arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if arm: return arm
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.armature_add(enter_editmode=True)
    arm = bpy.context.active_object; arm.name = "Armature"
    eb = arm.data.edit_bones
    spine = eb[0]; spine.name="spine"; spine.head=(0,0,0.9); spine.tail=(0,0,1.4)
    head = eb.new("head"); head.head=spine.tail; head.tail=(0,0,1.7)
    l_up = eb.new("upper_arm.L"); l_up.head=(0.05,0,1.35); l_up.tail=(0.35,0,1.35)
    r_up = eb.new("upper_arm.R"); r_up.head=(-0.05,0,1.35); r_up.tail=(-0.35,0,1.35)
    l_fk = eb.new("forearm.L");  l_fk.head=l_up.tail; l_fk.tail=(0.55,0,1.30)
    r_fk = eb.new("forearm.R");  r_fk.head=r_up.tail; r_fk.tail=(-0.55,0,1.30)
    to_object_mode()
    bpy.context.view_layer.objects.active = body_obj
    body_obj.select_set(True); arm.select_set(True)
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    body_obj.select_set(False); arm.select_set(False)
    return arm

def apply_pose_from_angles(arm, angles: dict):
    if not angles: return
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    pb = arm.pose.bones
    def rot_deg(name, axis, deg):
        if name not in pb or deg is None: return
        b = pb[name]; b.rotation_mode='XYZ'; e=list(b.rotation_euler)
        ax={"X":0,"Y":1,"Z":2}[axis]; d=max(-120.0, min(120.0, float(deg)))
        e[ax]=math.radians(d); b.rotation_euler=e
    rot_deg("upper_arm.L","Z",  args := angles.get("left_shoulder_abd"))
    rot_deg("upper_arm.R","Z", -angles.get("right_shoulder_abd", 0.0))
    rot_deg("forearm.L", "Y", -(180.0 - angles.get("left_elbow", 180.0)))
    rot_deg("forearm.R", "Y",  (180.0 - angles.get("right_elbow", 180.0)))
    rot_deg("head",      "Z",  angles.get("head_yaw"))
    to_object_mode()

# -------------------------- Scene & Model --------------------------
print("\n" + "="*80); print("[DEFORM] Setting up scene/model…"); print("="*80)
obj = _main_mesh(); obj.name = "Avatar"
bpy.context.view_layer.objects.active = obj
_set_simple_shape(obj); _ensure_uv(obj)

# -------------------------- Material & Textures --------------------------
print("\n" + "="*80); print("[DEFORM] Building projection material…"); print("="*80)
mat = bpy.data.materials.get("AvatarProjection") or bpy.data.materials.new("AvatarProjection")
mat.use_nodes = True; nodes = mat.node_tree.nodes; links = mat.node_tree.links; nodes.clear()
out = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

frontN = _load_img_node(nodes, links, args.frontTex)
sideN  = _load_img_node(nodes, links, args.sideTex)
backN  = _load_img_node(nodes, links, args.backTex)

front_nodes = _load_many(nodes, links, _split_list(args.frontTexList)) if args.frontTexList else ([frontN] if frontN else [])
side_nodes  = _load_many(nodes, links, _split_list(args.sideTexList))  if args.sideTexList  else ([sideN]  if sideN  else [])
back_nodes  = _load_many(nodes, links, _split_list(args.backTexList))  if args.backTexList  else ([backN]  if backN  else [])

front_color = _merge_images(nodes, links, front_nodes)
side_color  = _merge_images(nodes, links, side_nodes)
back_color  = _merge_images(nodes, links, back_nodes)

attr_node = nodes.new("ShaderNodeAttribute"); attr_node.attribute_name = _make_facemask_if_missing(obj)
if front_color and (side_color or back_color):
    base = side_color or back_color
    mix_face = nodes.new("ShaderNodeMixRGB"); mix_face.blend_type='MIX'
    links.new(attr_node.outputs["Fac"], mix_face.inputs["Fac"])
    links.new(base,    mix_face.inputs[1])
    links.new(front_color, mix_face.inputs[2])
    final_color = mix_face.outputs["Color"]
else:
    final_color = front_color or side_color or back_color or nodes.new("ShaderNodeRGB").outputs["Color"]
links.new(final_color, bsdf.inputs["Base Color"])

if not obj.data.materials: obj.data.materials.append(mat)
else: obj.data.materials[0] = mat

# -------------------------- Bake --------------------------
print("\n" + "="*80); print("[DEFORM] Baking…"); print("="*80)
to_object_mode()
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True); bpy.context.view_layer.objects.active = obj

tex_res = args.texRes if not args.highDetail else max(args.texRes, 4096)
print(f"[DEFORM] Bake resolution: {tex_res}x{tex_res}")
bake_img = bpy.data.images.new("BakedTexture", width=tex_res, height=tex_res, alpha=True)
bake_img.colorspace_settings.name = 'sRGB'

img_tex = nodes.new("ShaderNodeTexImage"); img_tex.image = bake_img
for n in nodes: n.select = False
img_tex.select = True; nodes.active = img_tex  # CRITICAL: active image for bake

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 64
bpy.context.scene.render.bake.use_pass_direct = False
bpy.context.scene.render.bake.use_pass_indirect = False
bpy.context.scene.render.bake.use_pass_color = True
bpy.context.scene.render.bake.margin = 16

print("[DEFORM] Baking diffuse…")
t0 = time.time(); bpy.ops.object.bake(type='DIFFUSE'); print(f"[DEFORM] Bake time: {time.time()-t0:.1f}s")

# Save → reload → pack (ensures embedding in GLB)
png_path = os.path.join(out_dir, "baked_texture.png")
bake_img.filepath_raw = png_path; bake_img.file_format = 'PNG'; bake_img.save()
bpy.data.images.remove(bake_img)         # drop in-mem
bake_img = bpy.data.images.load(png_path) # reload from disk
bake_img.name = "BakedTexture"; bake_img.pack()

# Switch material to use baked image via UV
nodes.clear()
out2 = nodes.new("ShaderNodeOutputMaterial")
bsdf2 = nodes.new("ShaderNodeBsdfPrincipled")
uv = nodes.new("ShaderNodeTexCoord")
t2 = nodes.new("ShaderNodeTexImage"); t2.image = bake_img
links.new(uv.outputs["UV"], t2.inputs["Vector"])
links.new(t2.outputs["Color"], bsdf2.inputs["Base Color"])
links.new(bsdf2.outputs["BSDF"], out2.inputs["Surface"])

# -------------------------- Pose (after bake) --------------------------
if args.poseJson and os.path.exists(args.poseJson):
    try:
        with open(args.poseJson) as f: pose_data = json.load(f)
        arm = ensure_basic_armature(obj); apply_pose_from_angles(arm, pose_data)
    except Exception as e:
        print(f"[DEFORM] Pose apply warning: {e}")

# -------------------------- Export --------------------------
def _prepare_selection_for_export():
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    for o in bpy.data.objects:
        if o.type == 'ARMATURE': o.select_set(True)
    bpy.context.view_layer.objects.active = obj

def _export_glb(path):
    to_object_mode(); _prepare_selection_for_export()
    Path(os.path.dirname(path) or "/tmp").mkdir(parents=True, exist_ok=True)

    # pack any loose image nodes (belt-and-suspenders)
    if obj.data.materials and obj.data.materials[0].use_nodes:
        for n in obj.data.materials[0].node_tree.nodes:
            if n.type == 'TEX_IMAGE' and n.image and not n.image.packed_file:
                n.image.pack()

    print(f"[DEFORM] Exporting GLB → {path}")
    res = bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        use_selection=True,
        export_texcoords=True,
        export_normals=True,
        export_colors=True,
        export_materials='EXPORT',
        export_image_format='AUTO',
        export_cameras=False,
        export_lights=False,
        export_animations=False,
        export_apply=False
    )
    time.sleep(0.2)
    if not os.path.exists(path):
        raise RuntimeError(f"GLB export reported {res} but file not found at {path}")
    size = os.path.getsize(path)
    if size == 0: raise RuntimeError("GLB created but is empty")
    print(f"[DEFORM] Export OK, size {size/1024/1024:.2f} MB")
    with open(path, "rb") as f:
        if f.read(4) != b"glTF":
            print("[DEFORM] Warning: GLB header not found (still may be valid).")
    return size

print(f"[DEFORM] Final output path: {OUTPUT_GLTF}")
try:
    _export_glb(OUTPUT_GLTF)
    print("[DEFORM] ✓ SUCCESS")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[DEFORM] ✗ EXPORT FAILED: {e}"); sys.exit(1)
