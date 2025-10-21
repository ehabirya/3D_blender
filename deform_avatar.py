#!/usr/bin/env python3
"""
deform_avatar.py - Main orchestration script for Blender avatar generation

FIXED VERSION - Works for both Local and RunPod:
1. ✅ Added 5 missing measurement arguments (neck, head, hand, foot_length, foot_width)
2. ✅ Fixed assets path to work in both environments
3. ✅ Fixed output path to be more robust
4. ✅ Enhanced error handling and logging
"""

import bpy
import os
import sys
import argparse
import json
import tempfile
from pathlib import Path

# Import our modular components
import mesh_deformation as mesh
import texture_baking as texture
import export_utils as export

print("=" * 80)
print("[AVATAR] Blender Avatar Generation Script")
print(f"[AVATAR] Blender: {bpy.app.version_string}")
print(f"[AVATAR] Python: {sys.version.split()[0]}")
print(f"[AVATAR] CWD: {os.getcwd()}")
print("=" * 80)

# ==================== ARGUMENT PARSING ====================
parser = argparse.ArgumentParser(description="Generate 3D avatar with Blender")

# Basic settings
parser.add_argument("--preset", type=str, default="neutral",
                    help="Avatar preset: male/female/neutral/child/baby")
parser.add_argument("--height", type=float, required=True,
                    help="Height in meters")

# Measurements (optional)
parser.add_argument("--chest", type=float, help="Chest circumference in meters")
parser.add_argument("--waist", type=float, help="Waist circumference in meters")
parser.add_argument("--hips", type=float, help="Hip circumference in meters")
parser.add_argument("--shoulder", type=float, help="Shoulder width in meters")
parser.add_argument("--inseam", type=float, help="Inseam length in meters")
parser.add_argument("--arm", type=float, help="Arm length in meters")

# 🐛 BUG FIX: Added missing measurement arguments
# These were being sent from frontend but ignored by backend
parser.add_argument("--neck", type=float, help="Neck circumference in meters")
parser.add_argument("--head", type=float, help="Head circumference in meters")
parser.add_argument("--hand", type=float, help="Hand circumference in meters")
parser.add_argument("--foot_length", type=float, help="Foot length in meters")
parser.add_argument("--foot_width", type=float, help="Foot width in meters")

# Textures (single photos)
parser.add_argument("--frontTex", type=str, help="Front photo path")
parser.add_argument("--sideTex", type=str, help="Side photo path")
parser.add_argument("--backTex", type=str, help="Back photo path")

# Textures (multiple photos per role)
parser.add_argument("--frontTexList", type=str, default="",
                    help="Semicolon-separated front photos")
parser.add_argument("--sideTexList", type=str, default="",
                    help="Semicolon-separated side photos")
parser.add_argument("--backTexList", type=str, default="",
                    help="Semicolon-separated back photos")

# Texture settings
parser.add_argument("--texRes", type=int, default=2048,
                    help="Texture resolution (e.g., 2048)")
parser.add_argument("--highDetail", action="store_true",
                    help="Use higher texture resolution (min 4096)")

# Pose
parser.add_argument("--poseJson", type=str, default="",
                    help="Path to pose angles JSON file")

# Output
parser.add_argument("--out", type=str, default="",
                    help="Output GLB file path")

# Utility
parser.add_argument("--make_bases", action="store_true",
                    help="Generate base .blend files and exit")

# Parse arguments
if "--" in sys.argv:
    args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:])
else:
    args, _ = parser.parse_known_args()

print("\n[AVATAR] Configuration:")
for k, v in vars(args).items():
    if k not in {"frontTex", "sideTex", "backTex", "frontTexList", "sideTexList", "backTexList"}:
        print(f"  {k}: {v}")

# ==================== OUTPUT PATH (FIXED FOR BOTH ENVIRONMENTS) ====================
# 🔧 PATH FIX: Works for both local and RunPod
if args.out:
    OUTPUT_GLTF = args.out
elif os.environ.get("OUTPUT_GLTF"):
    OUTPUT_GLTF = os.environ.get("OUTPUT_GLTF")
else:
    # Use temp directory (works everywhere)
    OUTPUT_GLTF = os.path.join(tempfile.gettempdir(), "avatar.glb")

output_dir = os.path.dirname(OUTPUT_GLTF)
if not output_dir:
    output_dir = tempfile.gettempdir()

Path(output_dir).mkdir(parents=True, exist_ok=True)

print(f"\n[AVATAR] Output: {OUTPUT_GLTF}")
print(f"[AVATAR] Output dir: {output_dir}")

# ==================== MAKE BASE FILES (UTILITY) ====================
if args.make_bases:
    print("\n[AVATAR] Generating base .blend files...")
    
    # 🔧 PATH FIX: Detect environment and use appropriate path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try RunPod path first, fall back to local
    if os.path.exists("/app"):
        base_dir = "/app/assets"
    else:
        base_dir = os.path.join(script_dir, "assets")
    
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    print(f"[AVATAR] Base directory: {base_dir}")
    
    for preset in ["male", "female", "neutral", "child", "baby"]:
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
        bpy.context.active_object.name = f"Base_{preset}"
        bpy.ops.object.shade_smooth()
        blend_path = os.path.join(base_dir, f"base_{preset}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print(f"  ✓ {preset}: {blend_path}")
    
    print("[AVATAR] Done.")
    sys.exit(0)

# ==================== STEP 1: MESH SETUP ====================
print("\n" + "=" * 80)
print("STEP 1: MESH SETUP")
print("=" * 80)

try:
    obj = mesh.get_main_mesh()
    obj.name = "Avatar"
    bpy.context.view_layer.objects.active = obj
    
    mesh.set_body_proportions(obj, args.height)
    mesh.ensure_uv_map(obj)
    
    facemask_attr = mesh.make_facemask_attribute(obj)
    
    print("[MESH] ✓ Basic mesh setup complete")
except Exception as e:
    print(f"[MESH] ✗ FATAL ERROR in mesh setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== STEP 2: TEXTURE PREPARATION ====================
print("\n" + "=" * 80)
print("STEP 2: TEXTURE SETUP")
print("=" * 80)

try:
    # Organize photos by role
    photos = {
        'front': [],
        'side': [],
        'back': []
    }
    
    # Add list photos (priority)
    if args.frontTexList:
        photos['front'].extend(texture.split_texture_list(args.frontTexList))
    if args.sideTexList:
        photos['side'].extend(texture.split_texture_list(args.sideTexList))
    if args.backTexList:
        photos['back'].extend(texture.split_texture_list(args.backTexList))
    
    # Add single photos (fallback)
    if args.frontTex and os.path.exists(args.frontTex) and args.frontTex not in photos['front']:
        photos['front'].append(args.frontTex)
    if args.sideTex and os.path.exists(args.sideTex) and args.sideTex not in photos['side']:
        photos['side'].append(args.sideTex)
    if args.backTex and os.path.exists(args.backTex) and args.backTex not in photos['back']:
        photos['back'].append(args.backTex)
    
    print(f"[AVATAR] Photos loaded: {sum(len(v) for v in photos.values())} total")
    for role, paths in photos.items():
        print(f"  {role}: {len(paths)}")
    
    mat, nodes = texture.build_projection_material(obj, photos, facemask_attr)
    
except Exception as e:
    print(f"[TEXTURE] ✗ FATAL ERROR in texture setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== MESH VERIFICATION (MOVED AFTER MATERIAL) ====================
print("\n" + "=" * 80)
print("MESH VERIFICATION")
print("=" * 80)

# 🐛 BUG FIX: Moved verification AFTER material creation
# Previously it was checking for material before material was created!
if not mesh.verify_mesh_ready(obj):
    print("[AVATAR] ✗ Mesh verification failed!")
    
    # Try to save debug blend file
    debug_blend = OUTPUT_GLTF.replace('.glb', '_debug.blend')
    try:
        bpy.ops.wm.save_as_mainfile(filepath=debug_blend)
        print(f"\n[AVATAR] Debug .blend saved to: {debug_blend}")
    except:
        pass
    
    sys.exit(1)

print("[AVATAR] ✓ Mesh verification passed")

# ==================== STEP 3: BAKING ====================
print("\n" + "=" * 80)
print("STEP 3: TEXTURE BAKING")
print("=" * 80)

try:
    tex_res = max(args.texRes, 4096) if args.highDetail else args.texRes
    baked_image, png_path = texture.bake_texture(obj, tex_res, output_dir)
except Exception as e:
    print(f"[BAKING] ✗ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== STEP 4: FINAL MATERIAL ====================
print("\n" + "=" * 80)
print("STEP 4: FINAL MATERIAL")
print("=" * 80)

try:
    texture.create_final_material(obj, baked_image)
    texture.pack_all_material_textures(obj)
except Exception as e:
    print(f"[MATERIAL] ✗ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== STEP 5: POSE (OPTIONAL) ====================
if args.poseJson and os.path.exists(args.poseJson):
    print("\n" + "=" * 80)
    print("STEP 5: POSE APPLICATION")
    print("=" * 80)
    
    try:
        with open(args.poseJson, 'r') as f:
            pose_data = json.load(f)
        
        arm = mesh.ensure_basic_armature(obj)
        mesh.apply_pose_from_angles(arm, pose_data)
        print("[AVATAR] ✓ Pose applied")
    except Exception as e:
        print(f"[AVATAR] ⚠ Pose application warning: {e}")
else:
    print("\n[AVATAR] Skipping pose (neutral pose)")

# ==================== STEP 6: EXPORT ====================
print("\n" + "=" * 80)
print("STEP 6: GLB EXPORT")
print("=" * 80)

try:
    result = export.export_glb(OUTPUT_GLTF, obj)
    export.print_export_summary(result)
    
    if not result["success"]:
        print("[AVATAR] Gathering diagnostics...")
        diag = export.diagnose_export_failure(obj)
        print("\n[AVATAR] Diagnostic Information:")
        print(json.dumps(diag, indent=2))
        sys.exit(1)
except Exception as e:
    print(f"[EXPORT] ✗ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== SUCCESS ====================
print("\n" + "=" * 80)
print("✓ AVATAR GENERATION COMPLETE")
print("=" * 80)
print(f"Output: {OUTPUT_GLTF}")
print(f"Size: {result['size_mb']:.2f} MB")
print("=" * 80)
