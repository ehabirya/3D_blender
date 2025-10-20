#!/usr/bin/env python3
"""
mesh_deformation_fixed.py

Add these functions to your mesh_deformation.py file to fix the verification issues.
"""

import bpy
import bmesh


def verify_mesh_ready(obj) -> bool:
    """
    IMPROVED VERSION: Verify mesh is ready for export with detailed logging.
    
    This replaces the existing verify_mesh_ready() function.
    
    Returns:
        bool: True if mesh is valid, False otherwise
    """
    print("\n[MESH_VERIFY] Starting mesh verification...")
    
    if not obj or not obj.data:
        print("[MESH_VERIFY] ✗ Object or mesh data is None")
        return False
    
    me = obj.data
    errors = []
    warnings = []
    
    # Check 1: Vertices
    vert_count = len(me.vertices)
    print(f"[MESH_VERIFY] Vertices: {vert_count}")
    if vert_count == 0:
        errors.append("No vertices")
        print("[MESH_VERIFY] ✗ No vertices in mesh")
        return False
    
    # Check 2: Faces
    face_count = len(me.polygons)
    print(f"[MESH_VERIFY] Faces: {face_count}")
    if face_count == 0:
        errors.append("No faces")
        print("[MESH_VERIFY] ✗ No faces in mesh")
        return False
    
    # Check 3: UV Map (CRITICAL for texture baking)
    if not me.uv_layers or len(me.uv_layers) == 0:
        errors.append("No UV map found")
        print("[MESH_VERIFY] ✗ No UV map - creating default...")
        
        # Auto-fix: Create UV map if missing
        try:
            if not me.uv_layers:
                me.uv_layers.new(name="UVMap")
                print("[MESH_VERIFY] ✓ Created default UV map")
            
            # Smart UV unwrap
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
            bpy.ops.object.mode_set(mode='OBJECT')
            print("[MESH_VERIFY] ✓ Auto-unwrapped UVs")
        except Exception as e:
            print(f"[MESH_VERIFY] ✗ Failed to create UV map: {e}")
            return False
    else:
        print(f"[MESH_VERIFY] ✓ UV layers: {len(me.uv_layers)}")
    
    # Check 4: Materials
    if not me.materials or len(me.materials) == 0:
        warnings.append("No materials assigned")
        print("[MESH_VERIFY] ⚠ No materials - will create default")
        
        # Auto-fix: Create basic material
        mat = bpy.data.materials.new(name="DefaultMaterial")
        mat.use_nodes = True
        if len(me.materials) == 0:
            me.materials.append(mat)
        else:
            me.materials[0] = mat
        print("[MESH_VERIFY] ✓ Created default material")
    else:
        print(f"[MESH_VERIFY] Materials: {len(me.materials)}")
        for i, mat in enumerate(me.materials):
            if mat is None:
                errors.append(f"Material slot {i} is empty")
                print(f"[MESH_VERIFY] ✗ Material slot {i} is empty")
            else:
                print(f"[MESH_VERIFY]   [{i}] {mat.name} (nodes: {mat.use_nodes})")
    
    # Check 5: Degenerate geometry
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(me)
    
    degenerate_count = 0
    for face in bm.faces:
        if face.calc_area() < 1e-6:
            degenerate_count += 1
    
    if degenerate_count > 0:
        warnings.append(f"{degenerate_count} degenerate faces")
        print(f"[MESH_VERIFY] ⚠ Found {degenerate_count} degenerate faces")
        
        # Auto-fix: Remove degenerate geometry
        try:
            bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-4)
            bmesh.update_edit_mesh(me)
            print("[MESH_VERIFY] ✓ Dissolved degenerate geometry")
        except Exception as e:
            print(f"[MESH_VERIFY] ⚠ Could not dissolve degenerate geometry: {e}")
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Check 6: Non-manifold geometry
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    
    # Get selected count
    bm = bmesh.from_edit_mesh(me)
    non_manifold_verts = [v for v in bm.verts if v.select]
    non_manifold_count = len(non_manifold_verts)
    
    if non_manifold_count > 0:
        warnings.append(f"{non_manifold_count} non-manifold vertices")
        print(f"[MESH_VERIFY] ⚠ Found {non_manifold_count} non-manifold vertices")
        
        # Don't auto-fix non-manifold - just warn
        # Non-manifold geometry can sometimes be intentional
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Check 7: Normals
    has_custom_normals = me.has_custom_normals
    print(f"[MESH_VERIFY] Custom normals: {has_custom_normals}")
    
    # Summary
    print(f"\n[MESH_VERIFY] Verification complete:")
    print(f"[MESH_VERIFY]   Errors: {len(errors)}")
    print(f"[MESH_VERIFY]   Warnings: {len(warnings)}")
    
    if errors:
        print("[MESH_VERIFY] ERRORS:")
        for err in errors:
            print(f"[MESH_VERIFY]   ✗ {err}")
        return False
    
    if warnings:
        print("[MESH_VERIFY] WARNINGS:")
        for warn in warnings:
            print(f"[MESH_VERIFY]   ⚠ {warn}")
    
    print("[MESH_VERIFY] ✓ Mesh is ready for export")
    return True


def ensure_uv_map(obj):
    """
    IMPROVED VERSION: Ensure mesh has a UV map.
    
    Creates and unwraps if missing.
    """
    me = obj.data
    
    if not me.uv_layers or len(me.uv_layers) == 0:
        print("[MESH] Creating UV map...")
        me.uv_layers.new(name="UVMap")
        
        # Unwrap
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
        print("[MESH] ✓ UV map created and unwrapped")
    else:
        print(f"[MESH] ✓ UV map exists: {me.uv_layers[0].name}")


def clean_mesh(obj):
    """
    Clean mesh geometry to prevent export issues.
    
    Removes:
    - Degenerate faces (zero area)
    - Loose vertices
    - Duplicate vertices
    - Zero-length edges
    """
    print("[MESH] Cleaning mesh geometry...")
    
    me = obj.data
    bpy.context.view_layer.objects.active = obj
    
    # Switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(me)
    
    initial_verts = len(bm.verts)
    initial_faces = len(bm.faces)
    
    # Remove degenerate geometry
    try:
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-4)
        print("[MESH]   ✓ Dissolved degenerate geometry")
    except Exception as e:
        print(f"[MESH]   ⚠ Could not dissolve degenerate: {e}")
    
    # Remove duplicate vertices
    try:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
        print("[MESH]   ✓ Removed duplicate vertices")
    except Exception as e:
        print(f"[MESH]   ⚠ Could not remove doubles: {e}")
    
    # Delete loose vertices
    try:
        loose_verts = [v for v in bm.verts if not v.link_faces]
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')
        print(f"[MESH]   ✓ Removed {len(loose_verts)} loose vertices")
    except Exception as e:
        print(f"[MESH]   ⚠ Could not remove loose vertices: {e}")
    
    bmesh.update_edit_mesh(me)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    final_verts = len(me.vertices)
    final_faces = len(me.polygons)
    
    print(f"[MESH] Cleaning complete:")
    print(f"[MESH]   Vertices: {initial_verts} → {final_verts}")
    print(f"[MESH]   Faces: {initial_faces} → {final_faces}")


def diagnose_mesh_issues(obj):
    """
    Comprehensive mesh diagnosis for debugging.
    
    Returns dict with all mesh properties.
    """
    me = obj.data
    
    diag = {
        "name": obj.name,
        "type": obj.type,
        "vertices": len(me.vertices),
        "edges": len(me.edges),
        "faces": len(me.polygons),
        "uv_layers": len(me.uv_layers) if me.uv_layers else 0,
        "materials": len(me.materials),
        "vertex_groups": len(obj.vertex_groups),
    }
    
    # Check for issues
    issues = []
    
    if diag["vertices"] == 0:
        issues.append("No vertices")
    if diag["faces"] == 0:
        issues.append("No faces")
    if diag["uv_layers"] == 0:
        issues.append("No UV layers")
    if diag["materials"] == 0:
        issues.append("No materials")
    
    # Check degenerate faces
    degenerate = sum(1 for p in me.polygons if p.area < 1e-6)
    if degenerate > 0:
        issues.append(f"{degenerate} degenerate faces")
    
    diag["issues"] = issues
    
    return diag


# Example usage in your main script:
"""
# After creating the mesh
obj = get_main_mesh()

# Clean it first
clean_mesh(obj)

# Ensure UV map
ensure_uv_map(obj)

# Verify before export
if not verify_mesh_ready(obj):
    # Print diagnostics
    diag = diagnose_mesh_issues(obj)
    print(f"Mesh diagnosis: {diag}")
    sys.exit(1)
"""
