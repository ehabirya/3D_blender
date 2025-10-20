#!/usr/bin/env python3
"""
mesh_deformation.py - Mesh and armature manipulation functions

FIXED VERSION with:
- Enhanced ensure_uv_map() that actually verifies success
- Detailed verify_mesh_ready() with auto-fixing
- Better error messages and diagnostics

Handles:
- Mesh creation and UV mapping
- Armature generation and rigging
- Pose application
- Body proportions/scaling
- FaceMask attribute generation
"""

import bpy
import math
import numpy as np
from typing import Optional, Dict


def to_object_mode():
    """Safely switch to Object mode."""
    try:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass


def get_main_mesh():
    """Get or create the main mesh object."""
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        print("[MESH] No mesh found; creating UV sphere.")
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
        return bpy.context.active_object
    return meshes[0]


def ensure_uv_map(obj):
    """
    Ensure the mesh has UV coordinates with proper error handling.
    
    FIXED:
    - Actually creates UV layer if missing
    - Verifies the unwrap succeeded
    - Provides detailed error messages
    - Raises exception on failure instead of silent fail
    """
    to_object_mode()
    
    # Check if UV already exists
    if obj.data.uv_layers and len(obj.data.uv_layers) > 0:
        print(f"[MESH] ✓ UV map exists: '{obj.data.uv_layers[0].name}'")
        return
    
    print("[MESH] No UV map found, creating...")
    
    # Verify mesh has geometry
    vert_count = len(obj.data.vertices)
    face_count = len(obj.data.polygons)
    
    if vert_count == 0 or face_count == 0:
        error_msg = f"Cannot create UV map: mesh has no geometry (verts={vert_count}, faces={face_count})"
        print(f"[MESH] ✗ {error_msg}")
        raise ValueError(error_msg)
    
    print(f"[MESH] Mesh geometry: {vert_count} verts, {face_count} faces")
    
    try:
        # Create UV layer explicitly
        if not obj.data.uv_layers:
            print("[MESH] Creating UV layer...")
            uv_layer = obj.data.uv_layers.new(name="UVMap")
            print(f"[MESH] ✓ UV layer '{uv_layer.name}' created")
        
        # Set as active
        obj.data.uv_layers.active = obj.data.uv_layers[0]
        
        # Ensure object is active in context
        bpy.context.view_layer.objects.active = obj
        
        # Switch to edit mode for unwrapping
        print("[MESH] Switching to edit mode for UV unwrap...")
        bpy.ops.object.mode_set(mode='EDIT')
        
        # Select all geometry
        bpy.ops.mesh.select_all(action='SELECT')
        
        # Unwrap with smart project
        print("[MESH] Running smart UV project...")
        result = bpy.ops.uv.smart_project(
            angle_limit=66.0,
            island_margin=0.02,
            area_weight=0.0,
            correct_aspect=True,
            scale_to_bounds=False
        )
        
        print(f"[MESH] Smart project result: {result}")
        
        # Switch back to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        
    except Exception as e:
        # Make sure we're back in object mode
        try:
            if bpy.context.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except:
            pass
        
        error_msg = f"UV unwrap failed: {e}"
        print(f"[MESH] ✗ {error_msg}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(error_msg)
    
    # CRITICAL: Verify UV map was actually created and has data
    if not obj.data.uv_layers or len(obj.data.uv_layers) == 0:
        error_msg = "UV map verification failed: No UV layers exist after unwrap"
        print(f"[MESH] ✗ {error_msg}")
        raise RuntimeError(error_msg)
    
    uv_count = len(obj.data.uv_layers[0].data)
    print(f"[MESH] ✓ UV map created successfully: {uv_count} UV coordinates")
    
    if uv_count == 0:
        error_msg = "UV map verification failed: UV layer is empty"
        print(f"[MESH] ✗ {error_msg}")
        raise RuntimeError(error_msg)


def set_body_proportions(obj, height_m: float):
    """
    Apply basic body scaling based on height.
    
    Args:
        obj: The mesh object
        height_m: Target height in meters
    """
    scale = max(0.2, float(height_m) / 1.75)
    obj.scale = (scale, scale, scale)
    print(f"[MESH] Body scale set to {scale:.3f} (height: {height_m:.2f}m)")


def make_facemask_attribute(obj, name: str = "FaceMask") -> str:
    """
    Create a vertex attribute for face region masking.
    
    This attribute is used to blend front-facing textures with the body.
    Values: 0.0 (body) to 1.0 (face region)
    
    Args:
        obj: The mesh object
        name: Attribute name
        
    Returns:
        The attribute name
    """
    if obj.data.attributes.get(name):
        print(f"[MESH] FaceMask attribute '{name}' already exists")
        return name
    
    print("[MESH] Building FaceMask attribute...")
    attr = obj.data.attributes.new(name=name, type='FLOAT', domain='POINT')
    verts = obj.data.vertices
    
    if not verts:
        return name
    
    # Analyze mesh geometry
    zs = np.array([v.co.z for v in verts], dtype=np.float32)
    z_min, z_max = float(zs.min()), float(zs.max())
    z_low = z_min + 0.65 * (z_max - z_min)
    z_high = z_min + 0.95 * (z_max - z_min)
    
    obj.data.calc_normals()
    vals = np.zeros(len(verts), dtype=np.float32)
    
    # Calculate face mask values
    for i, v in enumerate(verts):
        ny = v.normal.y
        z = v.co.z
        
        # Head band (vertical position)
        head_band = 0.0
        if z >= z_low:
            head_band = min(1.0, max(0.0, (z - z_low) / max(1e-6, (z_high - z_low))))
        
        # Front-facing (normal direction)
        frontness = 1.0 if ny < -0.15 else 0.0
        
        vals[i] = float(head_band * frontness)
    
    # Normalize and apply gamma for smoother falloff
    if vals.max() > 0:
        vals = (vals / vals.max()) ** 0.7
    
    for i, d in enumerate(attr.data):
        d.value = float(vals[i])
    
    print(f"[MESH] ✓ FaceMask created: {len(verts)} vertices")
    return name


def ensure_basic_armature(body_obj):
    """
    Create a simple armature for posing if one doesn't exist.
    
    Args:
        body_obj: The body mesh to rig
        
    Returns:
        The armature object
    """
    arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if arm:
        print("[MESH] Using existing armature")
        return arm
    
    print("[MESH] Creating basic armature...")
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.armature_add(enter_editmode=True)
    arm = bpy.context.active_object
    arm.name = "Armature"
    
    eb = arm.data.edit_bones
    
    # Spine
    spine = eb[0]
    spine.name = "spine"
    spine.head = (0, 0, 0.9)
    spine.tail = (0, 0, 1.4)
    
    # Head
    head = eb.new("head")
    head.head = spine.tail
    head.tail = (0, 0, 1.7)
    
    # Left arm
    l_up = eb.new("upper_arm.L")
    l_up.head = (0.05, 0, 1.35)
    l_up.tail = (0.35, 0, 1.35)
    
    l_fk = eb.new("forearm.L")
    l_fk.head = l_up.tail
    l_fk.tail = (0.55, 0, 1.30)
    
    # Right arm
    r_up = eb.new("upper_arm.R")
    r_up.head = (-0.05, 0, 1.35)
    r_up.tail = (-0.35, 0, 1.35)
    
    r_fk = eb.new("forearm.R")
    r_fk.head = r_up.tail
    r_fk.tail = (-0.55, 0, 1.30)
    
    to_object_mode()
    
    # Parent mesh to armature
    bpy.context.view_layer.objects.active = body_obj
    body_obj.select_set(True)
    arm.select_set(True)
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    body_obj.select_set(False)
    arm.select_set(False)
    
    print("[MESH] ✓ Armature created and rigged")
    return arm


def apply_pose_from_angles(arm, angles: Dict[str, Optional[float]]):
    """
    Apply pose to armature from angle dictionary.
    
    Args:
        arm: The armature object
        angles: Dictionary with keys like 'left_elbow', 'right_shoulder_abd', 'head_yaw'
    """
    if not angles:
        return
    
    print(f"[MESH] Applying pose with {len(angles)} angles...")
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='POSE')
    pb = arm.pose.bones
    
    def rot_deg(name: str, axis: str, deg: Optional[float]):
        """Set bone rotation in degrees."""
        if name not in pb or deg is None:
            return
        
        b = pb[name]
        b.rotation_mode = 'XYZ'
        e = list(b.rotation_euler)
        ax = {"X": 0, "Y": 1, "Z": 2}[axis]
        d = max(-120.0, min(120.0, float(deg)))
        e[ax] = math.radians(d)
        b.rotation_euler = e
    
    # Apply rotations
    rot_deg("upper_arm.L", "Z", angles.get("left_shoulder_abd"))
    rot_deg("upper_arm.R", "Z", -angles.get("right_shoulder_abd", 0.0))
    rot_deg("forearm.L", "Y", -(180.0 - angles.get("left_elbow", 180.0)))
    rot_deg("forearm.R", "Y", (180.0 - angles.get("right_elbow", 180.0)))
    rot_deg("head", "Z", angles.get("head_yaw"))
    
    to_object_mode()
    print("[MESH] ✓ Pose applied")


def verify_mesh_ready(obj) -> bool:
    """
    Verify mesh is ready for texture baking with detailed diagnostics and auto-fixing.
    
    FIXED VERSION with:
    - Shows WHICH check failed with details
    - Auto-fixes common issues (missing UV, empty material slots)
    - Provides actionable error messages
    - Returns detailed failure information
    
    Returns:
        True if ready, False otherwise
    """
    print("\n" + "=" * 80)
    print("[MESH_VERIFY] DETAILED MESH VERIFICATION")
    print("=" * 80)
    
    if not obj or not obj.data:
        print("[MESH_VERIFY] ✗ FATAL: Object or mesh data is None!")
        return False
    
    me = obj.data
    errors = []
    warnings = []
    
    # ===== CHECK 1: GEOMETRY =====
    print("\n[MESH_VERIFY] Check 1: Geometry")
    vert_count = len(me.vertices)
    face_count = len(me.polygons)
    edge_count = len(me.edges)
    
    print(f"[MESH_VERIFY]   Vertices: {vert_count}")
    print(f"[MESH_VERIFY]   Edges: {edge_count}")
    print(f"[MESH_VERIFY]   Faces: {face_count}")
    
    if vert_count == 0:
        errors.append("No vertices in mesh")
        print("[MESH_VERIFY]   ✗ No vertices!")
    elif face_count == 0:
        errors.append("No faces in mesh")
        print("[MESH_VERIFY]   ✗ No faces!")
    else:
        print("[MESH_VERIFY]   ✓ Geometry OK")
    
    # ===== CHECK 2: UV MAP =====
    print("\n[MESH_VERIFY] Check 2: UV Map")
    
    if not me.uv_layers or len(me.uv_layers) == 0:
        print("[MESH_VERIFY]   ✗ No UV layers found!")
        print("[MESH_VERIFY]   Auto-fixing: Creating UV map...")
        
        try:
            # Create UV layer
            uv_layer = me.uv_layers.new(name="UVMap")
            print(f"[MESH_VERIFY]   Created UV layer: '{uv_layer.name}'")
            
            # Unwrap
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # Verify
            if me.uv_layers and len(me.uv_layers) > 0:
                uv_count = len(me.uv_layers[0].data)
                print(f"[MESH_VERIFY]   ✓ UV map created: {uv_count} coordinates")
            else:
                errors.append("UV map creation failed")
                print("[MESH_VERIFY]   ✗ UV map creation FAILED")
                
        except Exception as e:
            errors.append(f"UV map creation error: {e}")
            print(f"[MESH_VERIFY]   ✗ Exception: {e}")
            
    else:
        uv_count = len(me.uv_layers[0].data)
        print(f"[MESH_VERIFY]   ✓ UV layers: {len(me.uv_layers)}")
        print(f"[MESH_VERIFY]   ✓ Active layer: '{me.uv_layers[0].name}'")
        print(f"[MESH_VERIFY]   ✓ UV coordinates: {uv_count}")
        
        if uv_count == 0:
            warnings.append("UV layer exists but has no coordinates")
            print("[MESH_VERIFY]   ⚠ UV layer is empty!")
    
    # ===== CHECK 3: MATERIALS =====
    print("\n[MESH_VERIFY] Check 3: Materials")
    
    if not me.materials or len(me.materials) == 0:
        print("[MESH_VERIFY]   ✗ No material slots!")
        print("[MESH_VERIFY]   Auto-fixing: Creating default material...")
        
        try:
            # Create basic material
            mat = bpy.data.materials.new(name="DefaultMaterial")
            mat.use_nodes = True
            
            # Assign to mesh
            me.materials.append(mat)
            
            print(f"[MESH_VERIFY]   ✓ Created material: '{mat.name}'")
            
        except Exception as e:
            errors.append(f"Material creation failed: {e}")
            print(f"[MESH_VERIFY]   ✗ Exception: {e}")
            
    else:
        print(f"[MESH_VERIFY]   Material slots: {len(me.materials)}")
        
        for i, mat in enumerate(me.materials):
            if mat is None:
                errors.append(f"Material slot {i} is empty (None)")
                print(f"[MESH_VERIFY]   ✗ Slot {i}: EMPTY (None)")
            else:
                node_info = f"nodes={mat.use_nodes}" if mat.use_nodes else "no nodes"
                print(f"[MESH_VERIFY]   ✓ Slot {i}: '{mat.name}' ({node_info})")
                
                # Check if material has texture nodes
                if mat.use_nodes:
                    tex_nodes = [n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE']
                    print(f"[MESH_VERIFY]       Texture nodes: {len(tex_nodes)}")
    
    # ===== CHECK 4: MESH QUALITY =====
    print("\n[MESH_VERIFY] Check 4: Mesh Quality")
    
    # Check for degenerate faces
    degenerate_count = sum(1 for p in me.polygons if p.area < 1e-6)
    if degenerate_count > 0:
        warnings.append(f"{degenerate_count} degenerate faces (area < 1e-6)")
        print(f"[MESH_VERIFY]   ⚠ Degenerate faces: {degenerate_count}")
    else:
        print(f"[MESH_VERIFY]   ✓ No degenerate faces")
    
    # Check for loose vertices
    loose_verts = sum(1 for v in me.vertices if len(v.link_edges) == 0)
    if loose_verts > 0:
        warnings.append(f"{loose_verts} loose vertices")
        print(f"[MESH_VERIFY]   ⚠ Loose vertices: {loose_verts}")
    else:
        print(f"[MESH_VERIFY]   ✓ No loose vertices")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 80)
    print("[MESH_VERIFY] VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"[MESH_VERIFY] Errors: {len(errors)}")
    print(f"[MESH_VERIFY] Warnings: {len(warnings)}")
    
    if errors:
        print("\n[MESH_VERIFY] ERRORS (blocking issues):")
        for i, err in enumerate(errors, 1):
            print(f"[MESH_VERIFY]   {i}. {err}")
    
    if warnings:
        print("\n[MESH_VERIFY] WARNINGS (non-blocking):")
        for i, warn in enumerate(warnings, 1):
            print(f"[MESH_VERIFY]   {i}. {warn}")
    
    success = len(errors) == 0
    
    if success:
        print("\n[MESH_VERIFY] ✓✓✓ MESH IS READY FOR BAKING ✓✓✓")
    else:
        print("\n[MESH_VERIFY] ✗✗✗ MESH VERIFICATION FAILED ✗✗✗")
        print("[MESH_VERIFY]")
        print("[MESH_VERIFY] Common causes:")
        print("[MESH_VERIFY]   • Base .blend file is corrupted or empty")
        print("[MESH_VERIFY]   • Measurements caused invalid mesh deformation")
        print("[MESH_VERIFY]   • texture.build_projection_material() didn't assign material")
        print("[MESH_VERIFY]   • UV unwrap failed due to bad geometry")
    
    print("=" * 80 + "\n")
    
    return success
