# app/deform_avatar.py
# Runs INSIDE Blender
# - Deform by height and (optionally) simple measurements
# - Project front/side/back photos in camera space
# - Blend by surface normal masks
# - Bake to a single UV texture
# - Export .glb with baked texture

import bpy, os, sys, math
from mathutils import Vector

# --------- arg parse ----------
import argparse
parser = argparse.ArgumentParser()
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
parser.add_argument("--texRes", type=int, default=2048)
parser.add_argument("--out", type=str, default="/tmp/avatar.glb")
parser.add_argument("--make_bases", action="store_true")  # optional one-time generator

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--")+1:]
else:
    argv = []
args = parser.parse_args(argv)

# ---------- optional: generate placeholder base blends ----------
if args.make_bases:
    base_dir = "/app/assets"
    os.makedirs(base_dir, exist_ok=True)
    for cat in ["male", "female", "neutral", "child", "baby"]:
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
        bpy.context.active_object.name = f"Base_{cat}"
        bpy.ops.object.shade_smooth()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(base_dir, f"base_{cat}.blend"))
    print("Base .blend files created.")
    sys.exit(0)

print("▶ starting deform + projection")

# ---------- pick main mesh ----------
mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
if not mesh_objs:
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
    obj = bpy.context.active_object
    obj.name = "Avatar"
else:
    obj = mesh_objs[0]
bpy.context.view_layer.objects.active = obj

# ---------- simple deformation ----------
# scale to requested height (assumes base ~1.75 m)
scale = max(0.2, float(args.height) / 1.75)
obj.scale = (scale, scale, scale)

# NOTE: you can expand with chest/waist/hips/shoulder shaping here.
# (e.g., use vertex groups or shape keys by name if your base has them)

# ---------- ensure UVs ----------
bpy.ops.object.mode_set(mode='OBJECT')
if not obj.data.uv_layers:
    # create a UV map; smart_project is robust for generic meshes
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=66)
    bpy.ops.object.mode_set(mode='OBJECT')

# ---------- create material for projection ----------
mat_name = "AvatarProjection"
mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(mat_name)
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

out = nodes.new("ShaderNodeOutputMaterial")
bsdf = nodes.new("ShaderNodeBsdfPrincipled")
links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

# texture coordinate & mapping for camera space
texcoord = nodes.new("ShaderNodeTexCoord")   # has 'Camera' and 'Window' outputs

# helpful geometry nodes
geom = nodes.new("ShaderNodeNewGeometry")    # for surface normal
separateXYZ = nodes.new("ShaderNodeSeparateXYZ")

links.new(geom.outputs["Normal"], separateXYZ.inputs["Vector"])

# load images if provided
def add_image_node(path):
    if not path or not os.path.exists(path): return None
    node = nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(path)
    node.projection = 'FLAT'
    # drive by camera-space coords
    links.new(texcoord.outputs["Camera"], node.inputs["Vector"])
    return node

frontN = add_image_node(args.frontTex)
sideN  = add_image_node(args.sideTex)
backN  = add_image_node(args.backTex)

# normal-based masks
# Blender Y axis points "back" in default front view; we want:
#  front mask ~ facing -Y, back mask ~ +Y, side mask ~ |X|
# use Normal.Y sign; mask = clamp(±Y, 0, 1); side uses abs(X)
clamp_front = nodes.new("ShaderNodeMath"); clamp_front.operation='MAXIMUM'
negY = nodes.new("ShaderNodeMath"); negY.operation='MULTIPLY'; negY.inputs[1].default_value = -1.0
links.new(separateXYZ.outputs["Y"], negY.inputs[0])         # -Y
links.new(negY.outputs["Value"], clamp_front.inputs[0])     # clamp with 0
clamp_front.inputs[1].default_value = 0.0

clamp_back = nodes.new("ShaderNodeMath"); clamp_back.operation='MAXIMUM'
links.new(separateXYZ.outputs["Y"], clamp_back.inputs[0])   # +Y
clamp_back.inputs[1].default_value = 0.0

absX = nodes.new("ShaderNodeMath"); absX.operation='ABSOLUTE'
links.new(separateXYZ.outputs["X"], absX.inputs[0])
# normalize side strength a bit
side_gain = nodes.new("ShaderNodeMath"); side_gain.operation='MULTIPLY'; side_gain.inputs[1].default_value = 0.8
links.new(absX.outputs["Value"], side_gain.inputs[0])

# Mix textures with masks (front vs side vs back)
# front/side first
mix_front_side = nodes.new("ShaderNodeMixRGB"); mix_front_side.blend_type='MIX'
mix_front_side.inputs["Fac"].default_value = 0.5

# if nodes exist, wire them; otherwise fall through to BSDF base color
color_socket = bsdf.inputs["Base Color"]

def link_if_tex(tex_node, dest_socket):
    if tex_node:
        links.new(tex_node.outputs["Color"], dest_socket)
        return True
    return False

wired = False
if frontN and sideN:
    # Factor = normalized front_mask / (front_mask + side_mask)
    sum_fs = nodes.new("ShaderNodeMath"); sum_fs.operation='ADD'
    links.new(clamp_front.outputs["Value"], sum_fs.inputs[0])
    links.new(side_gain.outputs["Value"],  sum_fs.inputs[1])

    div_f = nodes.new("ShaderNodeMath"); div_f.operation='DIVIDE'
    links.new(clamp_front.outputs["Value"], div_f.inputs[0])
    links.new(sum_fs.outputs["Value"],    div_f.inputs[1])

    links.new(div_f.outputs["Value"], mix_front_side.inputs["Fac"])
    links.new(frontN.outputs["Color"], mix_front_side.inputs[1])
    links.new(sideN.outputs["Color"],  mix_front_side.inputs[2])
    wired = True
elif frontN:
    wired = link_if_tex(frontN, color_socket)
elif sideN:
    wired = link_if_tex(sideN, color_socket)

if backN:
    # Mix previous result with back using back weight vs (previous total + back)
    if wired:
        mix_with_back = nodes.new("ShaderNodeMixRGB"); mix_with_back.blend_type='MIX'
        sum_all = nodes.new("ShaderNodeMath"); sum_all.operation='ADD'
        links.new(clamp_back.outputs["Value"], sum_all.inputs[0])

        # approximate: reuse sum_fs or side_gain as "others"
        others = side_gain.outputs["Value"] if (frontN or sideN) else clamp_front.outputs["Value"]
        links.new(others, sum_all.inputs[1])

        div_b = nodes.new("ShaderNodeMath"); div_b.operation='DIVIDE'
        links.new(clamp_back.outputs["Value"], div_b.inputs[0])
        links.new(sum_all.outputs["Value"],    div_b.inputs[1])

        links.new(div_b.outputs["Value"], mix_with_back.inputs["Fac"])
        links.new(backN.outputs["Color"], mix_with_back.inputs[2])  # Color2 = back
        links.new(mix_front_side.outputs["Color"] if (frontN and sideN) else (frontN.outputs["Color"] if frontN else sideN.outputs["Color"]),
                  mix_with_back.inputs[1])

        links.new(mix_with_back.outputs["Color"], color_socket)
    else:
        wired = link_if_tex(backN, color_socket)

if not wired:
    bsdf.inputs["Base Color"].default_value = (0.75, 0.75, 0.75, 1.0)

# assign material
obj.data.materials.clear()
obj.data.materials.append(mat)

# ---------- BAKE to single texture ----------
# Baking is required so the GLB has one proper UV map with pixels from our projections.
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'CPU'

# target bake image
img_name = "BakedTex"
bake_img = bpy.data.images.get(img_name) or bpy.data.images.new(img_name, width=int(args.texRes), height=int(args.texRes), alpha=True)
bake_node = nodes.new("ShaderNodeTexImage")
bake_node.image = bake_img
# Make it the active image node for baking
nodes.active = bake_node

# ensure an active UV map for baking
if not obj.data.uv_layers:
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=66)
    bpy.ops.object.mode_set(mode='OBJECT')

# Bake only color (no lighting)
bpy.context.view_layer.objects.active = obj
for o in bpy.context.selected_objects: o.select_set(False)
obj.select_set(True)

bake_settings = bpy.context.scene.render
bpy.context.scene.cycles.bake_type = 'DIFFUSE'
bpy.context.scene.render.use_bake_multires = False
bpy.context.scene.render.bake.use_pass_direct = False
bpy.context.scene.render.bake.use_pass_indirect = False
bpy.context.scene.render.bake.margin = 4

bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'})

# Replace procedural graph with baked texture for export
nodes.clear()
out2 = nodes.new("ShaderNodeOutputMaterial")
bsdf2 = nodes.new("ShaderNodeBsdfPrincipled")
links.new(bsdf2.outputs["BSDF"], out2.inputs["Surface"])
tex_baked = nodes.new("ShaderNodeTexImage")
tex_baked.image = bake_img
links.new(tex_baked.outputs["Color"], bsdf2.inputs["Base Color"])

# ---------- Export GLB ----------
os.makedirs(os.path.dirname(args.out), exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=args.out,
    export_format='GLB',
    export_texcoords=True,
    export_normals=True,
    export_materials='EXPORT',
    export_image_format='AUTO',
    export_yup=True
)

print(f"✅ Exported GLB: {args.out}")
