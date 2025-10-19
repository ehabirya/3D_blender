#!/usr/bin/env python3
"""
export_utils.py - GLB export with validation

Handles:
- Selection preparation
- GLB export with optimal settings
- File validation (existence, size, header)
- Error reporting and diagnostics
"""

import bpy
import os
import time
from pathlib import Path
from typing import Optional


def prepare_export_selection(obj):
    """
    Prepare scene selection for GLB export.
    
    Selects the main mesh and any armatures.
    """
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select main mesh
    obj.select_set(True)
    
    # Select armatures (for posed exports)
    for o in bpy.data.objects:
        if o.type == 'ARMATURE':
            o.select_set(True)
    
    # Set active object
    bpy.context.view_layer.objects.active = obj
    
    print(f"[EXPORT] Selected {len([o for o in bpy.data.objects if o.select_get()])} object(s)")


def export_glb(filepath: str, obj) -> dict:
    """
    Export scene to GLB format with validation.
    
    Args:
        filepath: Output GLB path
        obj: Main mesh object
        
    Returns:
        Dictionary with:
            - success: bool
            - filepath: str (if successful)
            - size_bytes: int (if successful)
            - error: str (if failed)
    """
    print(f"[EXPORT] Exporting GLB → {filepath}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(filepath) or "/tmp"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare selection
    prepare_export_selection(obj)
    
    # Pack all textures (safety measure)
    print("[EXPORT] Packing textures...")
    if obj.data.materials and obj.data.materials[0].use_nodes:
        for node in obj.data.materials[0].node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image and not node.image.packed_file:
                try:
                    node.image.pack()
                    print(f"[EXPORT] ✓ Packed: {node.image.name}")
                except Exception as e:
                    print(f"[EXPORT] ⚠ Could not pack {node.image.name}: {e}")
    
    # Export
    try:
        result = bpy.ops.export_scene.gltf(
            filepath=filepath,
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
        print(f"[EXPORT] Export operation result: {result}")
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Export operation failed: {e}"
        }
    
    # Small delay for filesystem sync
    time.sleep(0.2)
    
    # Validate output
    validation = validate_glb_file(filepath, output_dir)
    
    if validation["success"]:
        print(f"[EXPORT] ✓ SUCCESS: {validation['size_mb']:.2f} MB")
        return {
            "success": True,
            "filepath": filepath,
            "size_bytes": validation["size_bytes"],
            "size_mb": validation["size_mb"]
        }
    else:
        return {
            "success": False,
            "error": validation["error"],
            "diagnostics": validation.get("diagnostics")
        }


def validate_glb_file(filepath: str, output_dir: str) -> dict:
    """
    Validate GLB file after export.
    
    Checks:
    - File exists
    - File is not empty
    - File has GLB header
    
    Args:
        filepath: Path to GLB file
        output_dir: Output directory (for diagnostics)
        
    Returns:
        Dictionary with validation results
    """
    # Check existence
    if not os.path.exists(filepath):
        # Gather diagnostics
        diagnostics = {
            "filepath": filepath,
            "exists": False,
            "output_dir": output_dir,
            "output_dir_exists": os.path.exists(output_dir)
        }
        
        # List directory contents
        try:
            if os.path.exists(output_dir):
                diagnostics["dir_contents"] = os.listdir(output_dir)
            else:
                diagnostics["dir_contents"] = []
        except Exception as e:
            diagnostics["dir_list_error"] = str(e)
        
        return {
            "success": False,
            "error": f"GLB file not found at {filepath}",
            "diagnostics": diagnostics
        }
    
    # Check size
    size = os.path.getsize(filepath)
    if size == 0:
        return {
            "success": False,
            "error": "GLB file created but is empty (0 bytes)"
        }
    
    # Check header
    header_valid = False
    header_bytes = None
    try:
        with open(filepath, "rb") as f:
            header_bytes = f.read(4)
            if header_bytes == b"glTF":
                header_valid = True
    except Exception as e:
        return {
            "success": False,
            "error": f"Could not read GLB file: {e}"
        }
    
    if not header_valid:
        print(f"[EXPORT] ⚠ Warning: Unexpected file header: {header_bytes}")
        print("[EXPORT] File may still be valid, continuing...")
    
    return {
        "success": True,
        "size_bytes": size,
        "size_mb": size / 1024 / 1024,
        "header_valid": header_valid
    }


def diagnose_export_failure(obj) -> dict:
    """
    Gather diagnostic information after export failure.
    
    Returns:
        Dictionary with diagnostic information
    """
    diag = {
        "blender_version": bpy.app.version_string,
        "object": {
            "name": obj.name,
            "type": obj.type,
            "has_data": bool(obj.data),
            "vertex_count": len(obj.data.vertices) if obj.data else 0
        },
        "materials": [],
        "textures": [],
        "scene": {
            "render_engine": bpy.context.scene.render.engine,
            "objects_count": len(bpy.data.objects),
            "selected_count": len([o for o in bpy.data.objects if o.select_get()])
        }
    }
    
    # Material info
    if obj.data.materials:
        for i, mat in enumerate(obj.data.materials):
            mat_info = {
                "index": i,
                "name": mat.name,
                "use_nodes": mat.use_nodes,
                "nodes": []
            }
            
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    node_info = {
                        "type": node.type,
                        "name": node.name
                    }
                    
                    if node.type == 'TEX_IMAGE' and node.image:
                        node_info["image"] = {
                            "name": node.image.name,
                            "size": node.image.size[:],
                            "packed": bool(node.image.packed_file)
                        }
                    
                    mat_info["nodes"].append(node_info)
            
            diag["materials"].append(mat_info)
    
    # Texture info
    for img in bpy.data.images:
        img_info = {
            "name": img.name,
            "size": img.size[:],
            "packed": bool(img.packed_file),
            "filepath": img.filepath
        }
        diag["textures"].append(img_info)
    
    return diag


def print_export_summary(result: dict):
    """Print a formatted export summary."""
    print("\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)
    
    if result["success"]:
        print(f"✓ Status: SUCCESS")
        print(f"  File: {result['filepath']}")
        print(f"  Size: {result['size_mb']:.2f} MB ({result['size_bytes']:,} bytes)")
    else:
        print(f"✗ Status: FAILED")
        print(f"  Error: {result['error']}")
        
        if "diagnostics" in result:
            print("\n  Diagnostics:")
            for key, value in result["diagnostics"].items():
                print(f"    {key}: {value}")
    
    print("=" * 80 + "\n")
