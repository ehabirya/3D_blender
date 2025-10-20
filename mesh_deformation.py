#!/usr/bin/env python3
"""
COMPLETE FIX for mesh_deformation.py

Replace your existing ensure_uv_map() and verify_mesh_ready() with these versions.
"""

import bpy
import bmesh


def ensure_uv_map(obj):
    """
    Ensure the mesh has UV coordinates with proper error handling.
    
    FIXES:
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


def verify_mesh_ready(obj) -> bool:
    """
    Verify mesh is ready for texture baking with detailed diagnostics and auto-fixing.
    
    IMPROVEMENTS:
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
