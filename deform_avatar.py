#!/usr/bin/env python3
"""
deform_avatar.py - Blender avatar generation script

This script runs inside Blender to:
1. Load base mesh and apply measurements
2. Create multi-photo projection material with FaceMask
3. Bake textures in neutral pose
4. Apply pose after baking (optional)
5. Export as GLB

ENHANCEMENTS:
- Robust output path handling with multiple fallbacks
- Detailed debug logging
- File creation verification
- Better error messages
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
print(f"[DEFORM] Working directory: {os.getcwd()}")
print("=" * 80)

# -------------------------- CLI Arguments --------------------------
parser = argparse.ArgumentParser(description="Generate 3D avatar with Blender")
parser.add_argument("--preset", type=str, default="neutral", 
                    help="Body preset: male, female, neutral, child, baby")
parser.add_argument("--height", type=float, required=True,
                    help="Height in meters")
parser.add_argument("--chest", type=float, help="Chest measurement")
parser.add_argument("--waist", type=float, help="Waist measurement")
parser.add_argument("--hips", type=float, help="Hips measurement")
parser.add_argument("--shoulder", type=float, help="Shoulder measurement")
parser.add_argument("--inseam", type=float, help="Inseam measurement")
parser.add_argument("--arm", type=float, help="Arm measurement")
parser.add_argument("--frontTex", type=str, help="Front texture path")
parser.add_argument("--sideTex", type=str, help="Side texture path")
parser.add_argument("--backTex", type=str, help="Back texture path")
parser.add_argument("--frontTexList", type=str, default="",
                    help="Semicolon-separated list of front textures")
parser.add_argument("--sideTexList", type=str, default="",
                    help="Semicolon-separated list of side textures")
parser.add_argument("--backTexList", type=str, default="",
                    help="Semicolon-separated list of back textures")
parser.add_argument("--texRes", type=int, default=2048,
                    help="Texture resolution (512-8192)")
parser.add_argument("--highDetail", action="store_true",
                    help="Force high detail (≥4K) baking")
parser.add_argument("--poseJson", type=str, default="",
                    help="JSON file with pose angles (applied after bake)")
parser.add_argument("--out", type=str, default="/tmp/avatar.glb",
                    help="Output GLB file path")
parser.add_argument("--make_bases", action="store_true",
                    help="Generate base .blend files and exit")

# Parse arguments (handle both standard and Blender's -- separator)
if "--" in sys.argv:
    args, unknown = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:])
else:
    args, unknown = parser.parse_known_args()

print("[DEFORM] Parsed arguments:")
for key, value in vars(args).items():
    if value and key not in ['frontTex', 'sideTex', 'backTex']:  # Don't log full paths
        print(f"  {key}: {value}")

# -------------------- Output Path Handling --------------------
# Multiple fallback mechanisms to ensure we get the right path:
# 1. Environment variable (set by wrapper)
# 2. Command line argument
# 3. Default

OUTPUT_GLTF = os.environ.get("OUTPUT_GLTF") or args.out or "/tmp/avatar.glb"

print("=" * 80)
print("[DEFORM] Output Path Configuration:")
print(f"  Environment OUTPUT_GLTF: {os.environ.get('OUTPUT_GLTF', 'NOT SET')}")
print(f"  Argument --out: {args.out}")
print(f"  Final OUTPUT_GLTF: {OUTPUT_GLTF}")
print("=" * 80)

# Create output directory
output_dir = os.path.dirname(OUTPUT_GLTF)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    print(f"[DEFORM] Output directory: {output_dir}")
    print(f"[DEFORM] Directory exists: {os.path.exists(output_dir)}")
    if os.path.exists(output_dir):
        print(f"[DEFORM] Directory writable: {os.access(output_dir, os.W_OK)}")

# -------------------------- Base .blend Generator --------------------------
if args.make_bases:
    print("[DEFORM] Generating base .blend files...")
    base_dir = "/app/assets"
    os.makedirs(base_dir, exist_ok=True)
    
    for category in ["male", "female", "neutral", "child", "baby"]:
        print(f"  Creating base_{category}.blend...")
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
        bpy.context.active_object.name = f"Base_{category}"
        bpy.ops.object.shade_smooth()
        blend_path = os.path.join(base_dir, f"base_{category}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print(f"  Saved: {blend_path}")
    
    print("[DEFORM] Base .blend files created successfully")
    sys.exit(0)

# -------------------------- Helper Functions --------------------------

def _main_mesh():
    """Get or create the main mesh object"""
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        print("[DEFORM] No mesh found, creating UV sphere")
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
        return bpy.context.active_object
    print(f"[DEFORM] Found mesh: {meshes[0].name}")
    return meshes[0]


def _ensure_uv(obj):
    """Ensure object has UV mapping"""
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    
    if not obj.data.uv_layers:
        print(f"[DEFORM] Creating UV map for {obj.name}")
        try:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.uv.smart_project(angle_limit=66)
            print("[DEFORM] UV map created successfully")
        finally:
            bpy.ops.object.mode_set(mode='OBJECT')
    else:
        print(f"[DEFORM] UV map already exists: {obj.data.uv_layers[0].name}")


def _load_img_node(nodes, links, path):
    """Load an image as a shader node"""
    if not path or not os.path.exists(path):
        if path:
            print(f"[DEFORM] Warning: Image not found: {path}")
        return None
    
    print(f"[DEFORM] Loading image: {os.path.basename(path)}")
    img = bpy.data.images.load(path)
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = 'CLIP'
    
    # Connect to camera projection
    tcoord = nodes.get("TexCoord") or nodes.new("ShaderNodeTexCoord")
    tcoord.name = "TexCoord"
    links.new(tcoord.outputs["Camera"], tex.inputs["Vector"])
    
    return tex


def _split_list(arg: str):
    """Split semicolon-separated path list and validate"""
    if not arg:
        return []
    paths = [p.strip() for p in arg.split(";") if p.strip()]
    valid_paths = [p for p in paths if os.path.exists(p)]
    if len(valid_paths) < len(paths):
        print(f"[DEFORM] Warning: {len(paths) - len(valid_paths)} paths not found")
    return valid_paths


def _load_many(nodes, links, paths):
    """Load multiple images as shader nodes"""
    out = []
    for p in paths:
        n = _load_img_node(nodes, links, p)
        if n:
            out.append(n)
    return out


def _merge_images(nodes, links, tex_nodes, weights=None):
    """Merge multiple texture nodes with optional weights"""
    if not tex_nodes:
        return None
    if len(tex_nodes) == 1:
        return tex_nodes[0].outputs["Color"]
    
    # Normalize weights
    if not weights or len(weights) != len(tex_nodes):
        weights = [1.0] * len(tex_nodes)
    s = sum(weights) or 1.0
    weights = [w/s for w in weights]
    
    print(f"[DEFORM] Merging {len(tex_nodes)} images with weights {weights}")
    
    # Build weighted mix tree
    acc_color = tex_nodes[0].outputs["Color"]
    acc_w = weights[0]
    
    for tex, w in zip(tex_nodes[1:], weights[1:]):
        mix = nodes.new("ShaderNodeMixRGB")
        denom = nodes.new("ShaderNodeMath")
        denom.operation = 'ADD'
        denom.inputs[0].default_value = acc_w
        denom.inputs[1].default_value = w
        
        div = nodes.new("ShaderNodeMath")
        div.operation = 'DIVIDE'
        links.new(denom.outputs["Value"], div.inputs[1])
        div.inputs[0].default_value = w
        
        links.new(div.outputs["Value"], mix.inputs["Fac"])
        links.new(acc_color, mix.inputs[1])
        links.new(tex.outputs["Color"], mix.inputs[2])
        
        acc_color = mix.outputs["Color"]
        acc_w += w
    
    return acc_color


def _make_facemask_if_missing(obj, name="FaceMask"):
    """Create face mask attribute for selective texture blending"""
    if obj.data.attributes.get(name):
        print(f"[DEFORM] FaceMask attribute already exists")
        return name
    
    print(f"[DEFORM] Creating FaceMask attribute...")
    attr = obj.data.attributes.new(name=name, type='FLOAT', domain='POINT')
    verts = obj.data.vertices
    
    if len(verts) == 0:
        return name
    
    # Calculate z-based head region
    zs = np.array([v.co.z for v in verts], dtype=np.float32)
    z_min, z_max = float(zs.min()), float(zs.max())
    z_low = z_min + 0.65 * (z_max - z_min)
    z_high = z_min + 0.95 * (z_max - z_min)
    
    obj.data.calc_normals()
    vals = np.zeros(len(verts), dtype=np.float32)
    
    for i, v in enumerate(verts):
        ny = v.normal.y
        z = v.co.z
        
        # Head band calculation
        head_band = 0.0
        if z >= z_low:
            head_band = min(1.0, max(0.0, (z - z_low) / max(1e-6, (z_high - z_low))))
        
        # Frontness check
        frontness = 1.0 if ny < -0.15 else 0.0
        
        vals[i] = float(head_band * frontness)
    
    # Normalize and apply power curve
    if vals.max() > 0:
        vals = (vals / vals.max()) ** 0.7
    
    # Write to attribute
    attr_data = obj.data.attributes[name].data
    for i, v in enumerate(attr_data):
        v.value = float(vals[i])
    
    print(f"[DEFORM] FaceMask created with {len(verts)} vertices")
    return name


def _set_simple_shape(obj):
    """Apply height-based scaling"""
    scale = max(0.2, float(args.height) / 1.75)
    obj.scale = (scale, scale, scale)
    print(f"[DEFORM] Applied scale: {scale:.3f} (height: {args.height}m)")


def ensure_basic_armature(body_obj):
    """Create and bind a minimal armature if none exists"""
    # Check for existing armature
    arm = None
    for o in bpy.data.objects:
        if o.type == 'ARMATURE':
            arm = o
            print(f"[DEFORM] Found existing armature: {arm.name}")
            break
    
    if not arm:
        print("[DEFORM] Creating new armature...")
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.armature_add(enter_editmode=True)
        arm = bpy.context.active_object
        arm.name = "Armature"
        
        # Create bones
        eb = arm.data.edit_bones
        spine = eb[0]
        spine.name = "spine"
        spine.head = (0, 0, 0.9)
        spine.tail = (0, 0, 1.4)
        
        head = eb.new("head")
        head.head = spine.tail
        head.tail = (0, 0, 1.7)
        
        l_up = eb.new("upper_arm.L")
        l_up.head = (0.05, 0, 1.35)
        l_up.tail = (0.35, 0, 1.35)
        
        r_up = eb.new("upper_arm.R")
        r_up.head = (-0.05, 0, 1.35)
        r_up.tail = (-0.35, 0, 1.35)
        
        l_fk = eb.new("forearm.L")
        l_fk.head = l_up.tail
        l_fk.tail = (0.55, 0, 1.30)
        
        r_fk = eb.new("forearm.R")
        r_fk.head = r_up.tail
        r_fk.tail = (-0.55, 0, 1.30)
        
        bpy.ops.object.mode_set(mode='OBJECT')
        
        # Parent body to armature with automatic weights
        print("[DEFORM] Binding mesh to armature...")
        bpy.context.view_layer.objects.active = body_obj
        body_obj.select_set(True)
        arm.select_set(True)
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')
        body_obj.select_set(False)
        arm.select_set(False)
        print("[DEFORM] Armature created and bound")
    
    return arm


def apply_pose_from_angles(arm, angles: dict):
    """Apply pose from angle dictionary"""
    if not angles:
        return
    
    print(f"[DEFORM] Applying pose with angles: {angles}")
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    pb = arm.pose.bones
    
    def rot_deg(name, axis, deg):
        if name not in pb or deg is None:
            return
        b = pb[name]
        b.rotation_mode = 'XYZ'
        e = list(b.rotation_euler)
        ax = {"X": 0, "Y": 1, "Z": 2}[axis]
        d = max(-120.0, min(120.0, float(deg)))
        e[ax] = math.radians(d)
        b.rotation_euler = e
        print(f"  {name}.{axis} = {d}°")
    
    rot_deg("upper_arm.L", "Z", angles.get("left_shoulder_abd"))
    rot_deg("upper_arm.R", "Z", -angles.get("right_shoulder_abd") if angles.get("right_shoulder_abd") else None)
    rot_deg("forearm.L", "Y", -(180.0 - angles.get("left_elbow", 180.0)))
    rot_deg("forearm.R", "Y", (180.0 - angles.get("right_elbow", 180.0)))
    rot_deg("head", "Z", angles.get("head_yaw"))
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print("[DEFORM] Pose applied successfully")


# -------------------------- Scene & Model Setup --------------------------
print("\n" + "=" * 80)
print("[DEFORM] Setting up scene and model...")
print("=" * 80)

obj = _main_mesh()
obj.name = "Avatar"
bpy.context.view_layer.objects.active = obj

_set_simple_shape(obj)
_ensure_uv(obj)

# -------------------------- Material & Texture Setup --------------------------
print("\n" + "=" * 80)
print("[DEFORM] Creating material and texture nodes...")
print("=" * 80)

mat_name = "AvatarProjection"
mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

# Create output and BSDF
out = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

# Load single textures
frontN = _load_img_node(nodes, links, args.frontTex)
sideN = _load_img_node(nodes, links, args.sideTex)
backN = _load_img_node(nodes, links, args.backTex)

# Load multi-photo lists
frontList = _split_list(args.frontTexList)
sideList = _split_list(args.sideTexList)
backList = _split_list(args.backTexList)

print(f"[DEFORM] Photo counts - Front: {len(frontList)}, Side: {len(sideList)}, Back: {len(backList)}")

# Prepare node lists (prioritize multi-photo lists)
front_nodes = _load_many(nodes, links, frontList) if frontList else ([frontN] if frontN else [])
side_nodes = _load_many(nodes, links, sideList) if sideList else ([sideN] if sideN else [])
back_nodes = _load_many(nodes, links, backList) if backList else ([backN] if backN else [])

# Merge multi-photos per role
front_color = _merge_images(nodes, links, front_nodes)
side_color = _merge_images(nodes, links, side_nodes)
back_color = _merge_images(nodes, links, back_nodes)

# Create FaceMask for selective blending
face_mask_attr = _make_facemask_if_missing(obj, "FaceMask")
attr_node = nodes.new("ShaderNodeAttribute")
attr_node.attribute_name = face_mask_attr

# Combine front/side/back with FaceMask
if front_color and (side_color or back_color):
    print("[DEFORM] Blending front with side/back using FaceMask")
    base = side_color or back_color
    mix_face = nodes.new("ShaderNodeMixRGB")
    mix_face.blend_type = 'MIX'
    links.new(attr_node.outputs["Fac"], mix_face.inputs["Fac"])
    links.new(base, mix_face.inputs[1])
    links.new(front_color, mix_face.inputs[2])
    final_color = mix_face.outputs["Color"]
elif front_color:
    print("[DEFORM] Using front texture only")
    final_color = front_color
elif side_color:
    print("[DEFORM] Using side texture only")
    final_color = side_color
elif back_color:
    print("[DEFORM] Using back texture only")
    final_color = back_color
else:
    print("[DEFORM] No textures provided, using default gray")
    rgb = nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
    final_color = rgb.outputs["Color"]

# Connect to BSDF
links.new(final_color, bsdf.inputs["Base Color"])

# Assign material to object
if not obj.data.materials:
    obj.data.materials.append(mat)
else:
    obj.data.materials[0] = mat

print("[DEFORM] Material setup complete")

# -------------------------- Baking (Neutral Pose) --------------------------
print("\n" + "=" * 80)
print("[DEFORM] Starting texture baking...")
print("=" * 80)

# CRITICAL FIX: Deselect everything first, then select ONLY the mesh
bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True)
bpy.context.view_layer.objects.active = obj

# Determine texture resolution
tex_res = args.texRes if not args.highDetail else max(args.texRes, 4096)
print(f"[DEFORM] Bake resolution: {tex_res}x{tex_res}")

# Create bake target image
bake_img = bpy.data.images.new("BakedTexture", width=tex_res, height=tex_res, alpha=True)
bake_img.colorspace_settings.name = 'sRGB'

# Add image texture node for baking
img_tex = nodes.new("ShaderNodeTexImage")
img_tex.image = bake_img
img_tex.select = True
nodes.active = img_tex

# Configure Cycles baking
bpy.context.scene.cycles.samples = 64
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.cycles.bake_type = 'DIFFUSE'
bpy.context.scene.render.bake.use_pass_direct = False
bpy.context.scene.render.bake.use_pass_indirect = False
bpy.context.scene.render.bake.use_pass_color = True

print("[DEFORM] Baking diffuse texture...")
start_time = time.time()
bpy.ops.object.bake(type='DIFFUSE')
bake_time = time.time() - start_time
print(f"[DEFORM] Baking complete in {bake_time:.1f}s")
# Pack the baked texture into the blend file
print("[DEFORM] Packing baked texture...")
baked_image = bpy.data.images.get("BakedTexture")
if baked_image:
    if not baked_image.packed_file:
        baked_image.pack()
        print("[DEFORM] ✓ Texture packed successfully")
    else:
        print("[DEFORM] ✓ Texture already packed")
else:
    # Switch material to use baked texture
print("[DEFORM] Switching to baked texture...")
nodes.clear()

# Create material output and BSDF
out2 = nodes.new("ShaderNodeOutputMaterial")
bsdf2 = nodes.new("ShaderNodeBsdfPrincipled")

# Create texture coordinate node (CRITICAL!)
uv_node = nodes.new("ShaderNodeTexCoord")

# Create texture node and connect to UV coordinates
tex2 = nodes.new("ShaderNodeTexImage")
tex2.image = bake_img

# CRITICAL: Connect UV coordinates to texture
links.new(uv_node.outputs["UV"], tex2.inputs["Vector"])

# Connect texture to material
links.new(tex2.outputs["Color"], bsdf2.inputs["Base Color"])
links.new(bsdf2.outputs["BSDF"], out2.inputs["Surface"])

print("[DEFORM] ✓ Baked texture connected with UV coordinates")

# -------------------------- Pose Application (After Bake) --------------------------
if args.poseJson and os.path.exists(args.poseJson):
    print("\n" + "=" * 80)
    print(f"[DEFORM] Loading pose from {args.poseJson}")
    print("=" * 80)
    try:
        with open(args.poseJson, 'r') as f:
            pose_data = json.load(f)
        arm = ensure_basic_armature(obj)
        apply_pose_from_angles(arm, pose_data)
    except Exception as e:
        print(f"[DEFORM] Warning: Failed to apply pose: {e}")

# -------------------------- GLB Export --------------------------
print("\n" + "=" * 80)
print("[DEFORM] Preparing for GLB export...")
print("=" * 80)

def _ensure_object_mode():
    """Ensure we're in object mode"""
    try:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
            print("[DEFORM] Switched to OBJECT mode")
    except Exception as e:
        print(f"[DEFORM] Mode switch warning: {e}")


def _prepare_selection_for_export():
    """Select objects for export"""
    bpy.ops.object.select_all(action='DESELECT')
    
    # Always include main mesh
    obj.select_set(True)
    print(f"[DEFORM] Selected mesh: {obj.name}")
    
    # Include armature if it exists
    for o in bpy.data.objects:
        if o.type == 'ARMATURE':
            o.select_set(True)
            print(f"[DEFORM] Selected armature: {o.name}")
    
    bpy.context.view_layer.objects.active = obj


def _export_glb(filepath: str):
    """Export scene as GLB with verification and texture support"""
    _ensure_object_mode()
    _prepare_selection_for_export()
    
    print(f"[DEFORM] Exporting to: {filepath}")
    print(f"[DEFORM] Directory: {os.path.dirname(filepath)}")
    print(f"[DEFORM] Filename: {os.path.basename(filepath)}")
    
    # Ensure parent directory exists
    parent = os.path.dirname(filepath)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
        print(f"[DEFORM] Created directory: {parent}")
    
    # Verify material and texture before export
    if obj.data.materials:
        mat = obj.data.materials[0]
        print(f"[DEFORM] Material: {mat.name}")
        if mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    print(f"[DEFORM] ✓ Found texture node with image: {node.image.name}")
                    # ENSURE IMAGE IS PACKED BEFORE EXPORT
                    if not node.image.packed_file:
                        print(f"[DEFORM] Packing image before export...")
                        node.image.pack()
                    print(f"[DEFORM] ✓ Image packed: {node.image.packed_file is not None}")
    
    # Perform export with FULL material and texture support
    try:
        result = bpy.ops.export_scene.gltf(
            filepath=filepath,  # FIXED: was output_path
            export_format='GLB',
            use_selection=True,  # Export selected objects only
            
            # Geometry
            export_texcoords=True,
            export_normals=True,
            export_colors=True,
            export_tangents=False,
            
            # Materials - CRITICAL SETTINGS
            export_materials='EXPORT',
            export_image_format='AUTO',
            
            # Camera/Lights
            export_cameras=False,
            export_lights=False,
            
            # Animation
            export_animations=False,
            
            # Compression
            export_draco_mesh_compression_enable=False,
            
            # Other
            export_apply=False
        )
        print(f"[DEFORM] Export operation result: {result}")
    except Exception as e:
        print(f"[DEFORM] ERROR during export: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Wait for filesystem to settle
    time.sleep(0.2)
    
    # Verify file was created
    if not os.path.exists(filepath):
        parent_dir = os.path.dirname(filepath)
        if os.path.exists(parent_dir):
            contents = os.listdir(parent_dir)
            print(f"[DEFORM] ERROR: File not created!")
            print(f"[DEFORM] Directory contents: {contents}")
        else:
            print(f"[DEFORM] ERROR: Parent directory doesn't exist: {parent_dir}")
        
        raise RuntimeError(f"GLB export reported {result} but file not found at {filepath}")
    
    # Verify file size
    file_size = os.path.getsize(filepath)
    print(f"[DEFORM] Export successful!")
    print(f"[DEFORM] File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    if file_size == 0:
        raise RuntimeError(f"GLB file created but is empty (0 bytes)")
    
    # Verify GLB signature
    with open(filepath, 'rb') as f:
        header = f.read(12)
        if len(header) >= 4:
            magic = header[0:4]
            if magic == b'glTF':
                print("[DEFORM] ✓ Valid GLB file signature detected")
            else:
                print(f"[DEFORM] ⚠️ Warning: Unexpected file signature: {magic}")
    
    return file_size

# Perform the export
print(f"[DEFORM] Final output path: {OUTPUT_GLTF}")
try:
    file_size = _export_glb(OUTPUT_GLTF)
    print("\n" + "=" * 80)
    print(f"[DEFORM] ✓ SUCCESS! Avatar exported to: {OUTPUT_GLTF}")
    print(f"[DEFORM] File size: {file_size} bytes")
    print("=" * 80)
except Exception as e:
    print("\n" + "=" * 80)
    print(f"[DEFORM] ✗ EXPORT FAILED: {e}")
    print("=" * 80)
    import traceback
    traceback.print_exc()
    sys.exit(1)
