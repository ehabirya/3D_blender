#!/usr/bin/env python3
"""
mesh_deformation.py - Mesh and armature manipulation functions

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
    """Ensure the mesh has UV coordinates."""
    to_object_mode()
    if not obj.data.uv_layers:
        print("[MESH] Creating UV map...")
        try:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(angle_limit=66)
        finally:
            to_object_mode()
        print("[MESH] ✓ UV map created")


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
    Verify mesh is ready for texture baking.
    
    Returns:
        True if ready, False otherwise
    """
    checks = {
        "Has UV map": bool(obj.data.uv_layers),
        "Has material": bool(obj.data.materials),
        "Has geometry": bool(obj.data.vertices),
    }
    
    all_ok = all(checks.values())
    
    if not all_ok:
        print("[MESH] ✗ Mesh verification failed:")
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
    else:
        print("[MESH] ✓ Mesh ready for baking")
    
    return all_ok
