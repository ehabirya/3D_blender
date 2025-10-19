#!/usr/bin/env python3
"""
texture_baking.py - Texture projection, baking, and material handling

Handles:
- Multi-photo projection material setup
- Texture merging and blending
- Cycles baking with proper settings
- Texture save/reload/pack cycle (CRITICAL for GLB embedding)
- Material cleanup and final setup
"""

import bpy
import os
import time
from typing import List, Optional, Tuple
from pathlib import Path


def split_texture_list(arg: str) -> List[str]:
    """Split semicolon-separated texture paths and filter existing files."""
    return [p for p in (arg or "").split(";") if p and os.path.exists(p)]


def load_image_node(nodes, links, path: str):
    """
    Load an image as a texture node with camera projection.
    
    Args:
        nodes: Material node tree nodes
        links: Material node tree links
        path: Path to image file
        
    Returns:
        The texture node or None if image doesn't exist
    """
    if not path or not os.path.exists(path):
        if path:
            print(f"[TEXTURE] Warning: Missing image: {path}")
        return None
    
    # Load image
    img = bpy.data.images.load(path)
    
    # Create texture node
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.extension = 'CLIP'
    
    # Create or get texture coordinate node
    tcoord = nodes.get("TexCoord")
    if not tcoord:
        tcoord = nodes.new("ShaderNodeTexCoord")
        tcoord.name = "TexCoord"
    
    # Connect camera projection
    links.new(tcoord.outputs["Camera"], tex.inputs["Vector"])
    
    return tex


def load_multiple_images(nodes, links, paths: List[str]) -> List:
    """Load multiple images as texture nodes."""
    return [n for p in paths if (n := load_image_node(nodes, links, p))]


def merge_texture_outputs(nodes, links, tex_nodes: List, weights: Optional[List[float]] = None):
    """
    Merge multiple texture outputs with weighted blending.
    
    Args:
        nodes: Material node tree nodes
        links: Material node tree links
        tex_nodes: List of texture nodes to merge
        weights: Optional weights for each texture (defaults to equal weighting)
        
    Returns:
        The final merged color output socket
    """
    if not tex_nodes:
        return None
    
    if len(tex_nodes) == 1:
        return tex_nodes[0].outputs["Color"]
    
    # Normalize weights
    weights = weights or [1.0] * len(tex_nodes)
    s = sum(weights) or 1.0
    weights = [w / s for w in weights]
    
    # Progressively blend textures
    acc_color = tex_nodes[0].outputs["Color"]
    acc_w = weights[0]
    
    for tex, w in zip(tex_nodes[1:], weights[1:]):
        # Create mix node
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = 'MIX'
        
        # Calculate blend factor
        denom = nodes.new("ShaderNodeMath")
        denom.operation = 'ADD'
        denom.inputs[0].default_value = acc_w
        denom.inputs[1].default_value = w
        
        div = nodes.new("ShaderNodeMath")
        div.operation = 'DIVIDE'
        links.new(denom.outputs["Value"], div.inputs[1])
        div.inputs[0].default_value = w
        
        # Connect
        links.new(div.outputs["Value"], mix.inputs["Fac"])
        links.new(acc_color, mix.inputs[1])
        links.new(tex.outputs["Color"], mix.inputs[2])
        
        acc_color = mix.outputs["Color"]
        acc_w += w
    
    return acc_color


def build_projection_material(obj, photos: dict, facemask_attr: str) -> Tuple[object, object]:
    """
    Build multi-photo projection material with face masking.
    
    Args:
        obj: The mesh object
        photos: Dictionary with 'front', 'side', 'back' photo paths/lists
        facemask_attr: Name of the face mask attribute
        
    Returns:
        Tuple of (material, nodes)
    """
    print("[TEXTURE] Building projection material...")
    
    mat = bpy.data.materials.get("AvatarProjection")
    if not mat:
        mat = bpy.data.materials.new("AvatarProjection")
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Create output and BSDF
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    
    # Load photos
    front_nodes = load_multiple_images(nodes, links, photos.get('front', []))
    side_nodes = load_multiple_images(nodes, links, photos.get('side', []))
    back_nodes = load_multiple_images(nodes, links, photos.get('back', []))
    
    # Merge each role's photos
    front_color = merge_texture_outputs(nodes, links, front_nodes)
    side_color = merge_texture_outputs(nodes, links, side_nodes)
    back_color = merge_texture_outputs(nodes, links, back_nodes)
    
    # Create face mask attribute node
    attr_node = nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = facemask_attr
    
    # Blend front with side/back using face mask
    if front_color and (side_color or back_color):
        base = side_color or back_color
        mix_face = nodes.new("ShaderNodeMixRGB")
        mix_face.blend_type = 'MIX'
        links.new(attr_node.outputs["Fac"], mix_face.inputs["Fac"])
        links.new(base, mix_face.inputs[1])
        links.new(front_color, mix_face.inputs[2])
        final_color = mix_face.outputs["Color"]
    else:
        final_color = front_color or side_color or back_color
        if not final_color:
            # Fallback: solid color
            rgb = nodes.new("ShaderNodeRGB")
            rgb.outputs["Color"].default_value = (0.8, 0.7, 0.6, 1.0)
            final_color = rgb.outputs["Color"]
    
    links.new(final_color, bsdf.inputs["Base Color"])
    
    # Apply material to object
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat
    
    photo_counts = {
        'front': len(front_nodes),
        'side': len(side_nodes),
        'back': len(back_nodes)
    }
    print(f"[TEXTURE] ✓ Projection material built: {photo_counts}")
    
    return mat, nodes


def bake_texture(obj, tex_res: int, output_dir: str) -> Tuple[object, str]:
    """
    Bake the projection material to a texture.
    
    CRITICAL: This function implements the save→reload→pack cycle
    that ensures the texture is properly embedded in the GLB export.
    
    Args:
        obj: The mesh object
        tex_res: Texture resolution (e.g., 2048)
        output_dir: Directory to save the baked texture
        
    Returns:
        Tuple of (baked_image, png_path)
    """
    print(f"[TEXTURE] Baking at {tex_res}x{tex_res}...")
    
    # Ensure object mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Select only the target object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # Create bake target image
    bake_img = bpy.data.images.new("BakedTexture", width=tex_res, height=tex_res, alpha=True)
    bake_img.colorspace_settings.name = 'sRGB'
    
    # Add image texture node for bake target
    mat = obj.data.materials[0]
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    img_tex = nodes.new("ShaderNodeTexImage")
    img_tex.image = bake_img
    
    # CRITICAL: Set as active node for baking
    for n in nodes:
        n.select = False
    img_tex.select = True
    nodes.active = img_tex
    
    # Configure Cycles for baking
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.render.bake.use_pass_direct = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color = True
    bpy.context.scene.render.bake.margin = 16
    
    # Bake
    print("[TEXTURE] Baking diffuse...")
    t0 = time.time()
    bpy.ops.object.bake(type='DIFFUSE')
    print(f"[TEXTURE] ✓ Bake completed in {time.time() - t0:.1f}s")
    
    # CRITICAL SECTION: Save → Remove → Reload → Pack
    # This ensures the texture is properly embedded in GLB
    
    png_path = os.path.join(output_dir, "baked_texture.png")
    
    print("[TEXTURE] Saving baked texture to disk...")
    bake_img.filepath_raw = png_path
    bake_img.file_format = 'PNG'
    bake_img.save()
    print(f"[TEXTURE] ✓ Saved to: {png_path}")
    
    # Remove in-memory version
    print("[TEXTURE] Removing in-memory texture...")
    bpy.data.images.remove(bake_img)
    
    # Reload from disk
    print("[TEXTURE] Reloading texture from disk...")
    bake_img = bpy.data.images.load(png_path)
    bake_img.name = "BakedTexture"
    
    # Pack into .blend file
    print("[TEXTURE] Packing texture...")
    bake_img.pack()
    packed_size = len(bake_img.packed_file.data) if bake_img.packed_file else 0
    print(f"[TEXTURE] ✓ Packed: {packed_size / 1024 / 1024:.2f} MB")
    
    return bake_img, png_path


def create_final_material(obj, baked_image):
    """
    Replace projection material with a simple UV-mapped material using the baked texture.
    
    CRITICAL: This ensures the GLB exporter can find and embed the texture.
    
    Args:
        obj: The mesh object
        baked_image: The baked and packed image
    """
    print("[TEXTURE] Creating final material...")
    
    mat = obj.data.materials[0]
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear all nodes
    nodes.clear()
    
    # Create simple material
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    
    # UV coordinate node
    uv_node = nodes.new("ShaderNodeTexCoord")
    
    # Texture node with baked image
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = baked_image
    tex_node.name = "BakedTexture"
    
    # CRITICAL: Connect UV to texture
    links.new(uv_node.outputs["UV"], tex_node.inputs["Vector"])
    
    # Connect texture to BSDF
    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    
    print("[TEXTURE] ✓ Final material created")
    
    # Verify setup
    verify_material_for_export(obj, baked_image)


def verify_material_for_export(obj, baked_image) -> bool:
    """
    Verify material is properly set up for GLB export.
    
    Returns:
        True if ready, False otherwise
    """
    checks = {
        "Has material": bool(obj.data.materials),
        "Material uses nodes": obj.data.materials[0].use_nodes if obj.data.materials else False,
        "Has texture node": False,
        "Texture has image": False,
        "Image is packed": False,
    }
    
    if checks["Material uses nodes"]:
        mat = obj.data.materials[0]
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                checks["Has texture node"] = True
                checks["Texture has image"] = True
                checks["Image is packed"] = bool(node.image.packed_file)
                break
    
    all_ok = all(checks.values())
    
    if not all_ok:
        print("[TEXTURE] ✗ Material verification failed:")
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
    else:
        print("[TEXTURE] ✓ Material ready for export")
    
    return all_ok


def pack_all_material_textures(obj):
    """
    Pack all textures in the object's materials (belt-and-suspenders).
    
    This is a safety measure to ensure no loose references remain.
    """
    if not obj.data.materials:
        return
    
    packed_count = 0
    for mat in obj.data.materials:
        if not mat.use_nodes:
            continue
        
        for node in mat.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image and not node.image.packed_file:
                try:
                    node.image.pack()
                    packed_count += 1
                except Exception as e:
                    print(f"[TEXTURE] Warning: Could not pack {node.image.name}: {e}")
    
    if packed_count > 0:
        print(f"[TEXTURE] ✓ Packed {packed_count} additional texture(s)")
