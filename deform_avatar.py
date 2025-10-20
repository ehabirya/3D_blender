#!/usr/bin/env python3
"""
deform_avatar_debug.py - Debug version with enhanced error capture

CHANGES:
1. Detailed mesh verification logging
2. Measurement validation and fallbacks
3. Better error reporting
"""

import bpy
import os
import sys
import argparse
import json
from pathlib import Path

# Import our modular components
import mesh_deformation as mesh
import texture_baking as texture
import export_utils as export

print("=" * 80)
print("[AVATAR] Blender Avatar Generation Script (DEBUG MODE)")
print(f"[AVATAR] Blender: {bpy.app.version_string}")
print(f"[AVATAR] Python: {sys.version.split()[0]}")
print(f"[AVATAR] CWD: {os.getcwd()}")
print("=" * 80)

# ==================== MEASUREMENT VALIDATION ====================
def validate_and_fix_measurements(args):
    """
    Validate measurements and apply fallbacks for unreasonable values.
    
    Returns: dict of validated measurements
    """
    print("\n[DEBUG] Validating measurements...")
    
    height = args.height
    validated = {}
    
    # Define reasonable ranges as % of height
    reasonable_ranges = {
        'chest': (0.45, 0.65),      # 45-65% of height
        'waist': (0.35, 0.55),      # 35-55% of height
        'hips': (0.40, 0.60),       # 40-60% of height
        'shoulder': (0.20, 0.30),   # 20-30% of height
        'inseam': (0.40, 0.48),     # 40-48% of height
        'arm': (0.36, 0.42),        # 36-42% of height
        'neck': (0.18, 0.25),       # 18-25% of height
        'head': (0.30, 0.38),       # 30-38% of height
        'foot_length': (0.13, 0.17), # 13-17% of height
        'foot_width': (0.04, 0.07),  # 4-7% of height
    }
    
    # Default values as % of height
    defaults = {
        'chest': 0.52,
        'waist': 0.42,
        'hips': 0.50,
        'shoulder': 0.25,
        'inseam': 0.45,
        'arm': 0.38,
        'neck': 0.20,
        'head': 0.35,
        'foot_length': 0.15,
        'foot_width': 0.055,
    }
    
    for measure in reasonable_ranges.keys():
        value = getattr(args, measure, None)
        
        if value is None:
            # Use default
            validated[measure] = height * defaults[measure]
            print(f"[DEBUG] {measure}: None → using default {validated[measure]:.3f}m")
            continue
        
        # Check if reasonable
        min_val, max_val = reasonable_ranges[measure]
        percentage = value / height
        
        if min_val <= percentage <= max_val:
            validated[measure] = value
            print(f"[DEBUG] {measure}: {value:.3f}m ✓ ({percentage*100:.1f}% of height)")
        else:
            # Use default
            validated[measure] = height * defaults[measure]
            print(f"[DEBUG] {measure}: {value:.3f}m ✗ UNREASONABLE ({percentage*100:.1f}% of height)")
            print(f"[DEBUG]   → Replacing with default {validated[measure]:.3f}m")
    
    return validated


# ==================== ENHANCED MESH VERIFICATION ====================
def verify_mesh_with_details(obj):
    """
    Enhanced mesh verification with detailed error reporting.
    
    Returns: (success: bool, errors: list)
    """
    errors = []
    warnings = []
    
    print("\n[DEBUG] Running detailed mesh verification...")
    
    # Check 1: Object has mesh data
    if not obj.data:
        errors.append("Object has no mesh data")
        return False, errors, warnings
    
    me = obj.data
    
    # Check 2: Vertex count
    vert_count = len(me.vertices)
    print(f"[DEBUG] Vertices: {vert_count}")
    if vert_count == 0:
        errors.append("Mesh has no vertices")
    elif vert_count < 8:
        warnings.append(f"Very low vertex count: {vert_count}")
    
    # Check 3: Face count
    face_count = len(me.polygons)
    print(f"[DEBUG] Faces: {face_count}")
    if face_count == 0:
        errors.append("Mesh has no faces")
    elif face_count < 4:
        warnings.append(f"Very low face count: {face_count}")
    
    # Check 4: UV map
    if not me.uv_layers or len(me.uv_layers) == 0:
        errors.append("Mesh has no UV map")
    else:
        print(f"[DEBUG] UV layers: {len(me.uv_layers)}")
        uv_layer = me.uv_layers[0]
        print(f"[DEBUG] UV layer name: {uv_layer.name}")
    
    # Check 5: Materials
    if not me.materials or len(me.materials) == 0:
        warnings.append("Mesh has no materials")
    else:
        print(f"[DEBUG] Materials: {len(me.materials)}")
        for i, mat in enumerate(me.materials):
            if mat is None:
                errors.append(f"Material slot {i} is empty")
            else:
                print(f"[DEBUG]   [{i}] {mat.name}: nodes={mat.use_nodes}")
    
    # Check 6: Degenerate geometry
    degenerate_faces = 0
    for poly in me.polygons:
        if poly.area < 1e-6:
            degenerate_faces += 1
    
    if degenerate_faces > 0:
        warnings.append(f"Found {degenerate_faces} degenerate faces (area < 1e-6)")
        print(f"[DEBUG] Degenerate faces: {degenerate_faces}")
    
    # Check 7: Loose vertices
    loose_verts = sum(1 for v in me.vertices if len(v.link_edges) == 0)
    if loose_verts > 0:
        warnings.append(f"Found {loose_verts} loose vertices")
        print(f"[DEBUG] Loose vertices: {loose_verts}")
    
    # Print summary
    print(f"\n[DEBUG] Verification summary:")
    print(f"[DEBUG]   Errors: {len(errors)}")
    print(f"[DEBUG]   Warnings: {len(warnings)}")
    
    if errors:
        print("[DEBUG] ERRORS:")
        for err in errors:
            print(f"[DEBUG]   ✗ {err}")
    
    if warnings:
        print("[DEBUG] WARNINGS:")
        for warn in warnings:
            print(f"[DEBUG]   ⚠ {warn}")
    
    success = len(errors) == 0
    return success, errors, warnings


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
parser.add_argument("--out", type=str, default="/tmp/avatar.glb",
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

# Validate measurements
validated_measurements = validate_and_fix_measurements(args)

# ==================== OUTPUT PATH ====================
OUTPUT_GLTF = os.environ.get("OUTPUT_GLTF") or args.out or "/tmp/avatar.glb"
output_dir = os.path.dirname(OUTPUT_GLTF) or "/tmp"
Path(output_dir).mkdir(parents=True, exist_ok=True)

print(f"\n[AVATAR] Output: {OUTPUT_GLTF}")
print(f"[AVATAR] Output dir: {output_dir}")

# ==================== MAKE BASE FILES (UTILITY) ====================
if args.make_bases:
    print("\n[AVATAR] Generating base .blend files...")
    base_dir = "/app/assets"
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
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

# ==================== ENHANCED MESH VERIFICATION ====================
print("\n" + "=" * 80)
print("MESH VERIFICATION (DETAILED)")
print("=" * 80)

success, errors, warnings = verify_mesh_with_details(obj)

if not success:
    print("\n[AVATAR] ✗ MESH VERIFICATION FAILED!")
    print("[AVATAR] Errors found:")
    for err in errors:
        print(f"[AVATAR]   • {err}")
    
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
        import json
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
